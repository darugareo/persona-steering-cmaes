import os
import json
import torch
import cma
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import time

# 高L2ペルソナ（L2 ≥ 10）
HIGH_L2_PERSONAS = [
    "episode-179307_A", "episode-175246_A", "episode-134226_A",
    "episode-136981_B", "episode-140544_B", "episode-118328_B",
    "episode-84804_A", "episode-19493_A", "episode-158821_B",
    "episode-16276_B", "episode-24275_A", "episode-74475_A",
    "episode-184019_A", "episode-128744_B"
]


def get_partner_role(speaker_role):
    role_pairs = {
        "Husband": "Wife", "Wife": "Husband",
        "Parent": "Child", "Child": "Parent",
        "Mentor": "Mentee", "Mentee": "Mentor",
    }
    return role_pairs.get(speaker_role, "Partner")


def get_relationship_type(relationship):
    relationship_map = {
        "Husband and Wife": "married couple",
        "Parent and Child": "parent-child",
        "Mentee and Mentor": "mentor-mentee",
        "Classmates": "classmates",
        "Neighbors": "neighbors",
    }
    return relationship_map.get(relationship, relationship.lower())


def build_prompt(turn, profile):
    speaker_role = profile["speaker_role"]
    partner_role = get_partner_role(speaker_role)
    relationship_type = get_relationship_type(profile["relationship"])

    prompt = f"""You are Speaker A in a conversation with Speaker B.

Speaker A role: {speaker_role}
Speaker B role: {partner_role}
Relationship: {relationship_type}

Your task:
Given the conversation so far and Speaker B's latest utterance,
produce a single natural reply as Speaker A.

Constraints:
- Respond only as Speaker A.
- Do not change roles or speak as Speaker B.
- Do not introduce facts not present in the conversation.
- The reply should be natural given the specified relationship.
- Output only the reply text.

Conversation so far:
{turn.get('context', '')}

Speaker B's latest message:
"{turn['user']}"

Your reply:"""

    return prompt


def load_persona_data(persona_id):
    persona_dir = Path(f"personas_cc/{persona_id}")

    with open(persona_dir / "profile.json") as f:
        profile = json.load(f)

    train_file = persona_dir / "train_turns_persona_specific.json"
    with open(train_file) as f:
        train_data = json.load(f)

    return profile, train_data["turns"]


def load_trait_vectors(trait_dir="data/steering_vectors_v2", layer=20):
    trait_vectors = []
    for i in range(1, 6):
        trait_file = Path(trait_dir) / f"R{i}" / f"layer{layer}_svd.pt"
        tv = torch.load(trait_file, map_location="cpu", weights_only=False)
        trait_vectors.append(tv)
    return trait_vectors


class SteeringOptimizer:
    def __init__(self, model, tokenizer, trait_vectors, target_layer=20, alpha=2.0, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        # Convert trait vectors to same dtype as model (float16)
        model_dtype = next(model.parameters()).dtype
        self.trait_vectors = [tv.to(device=device, dtype=model_dtype) for tv in trait_vectors]
        self.target_layer = target_layer
        self.alpha = alpha
        self.device = device
        self.steering_vector = None
        self.hook_handle = None

    def _steering_hook(self, module, input, output):
        if self.steering_vector is not None:
            # Output is tuple: (hidden_states, ...)
            hidden_states = output[0]
            # Reshape steering_vector to (1, 1, hidden_size) for broadcasting
            # Ensure steering vector matches hidden_states dtype
            steering_vec = self.steering_vector.view(1, 1, -1).to(dtype=hidden_states.dtype)
            steered_hidden_states = hidden_states + self.alpha * steering_vec
            # Return modified output as tuple
            if isinstance(output, tuple):
                return (steered_hidden_states,) + output[1:]
            else:
                return steered_hidden_states
        return output

    def compute_steering_vector(self, weights):
        sv = torch.zeros_like(self.trait_vectors[0])
        for w, tv in zip(weights, self.trait_vectors):
            # Convert weight to same dtype as trait vector
            w_tensor = torch.tensor(w, dtype=tv.dtype, device=tv.device)
            sv += w_tensor * tv
        return sv

    def generate_with_steering(self, prompt, weights, max_new_tokens=100):
        self.steering_vector = self.compute_steering_vector(weights)
        layer = self.model.model.layers[self.target_layer]
        self.hook_handle = layer.register_forward_hook(self._steering_hook)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()
        finally:
            self.hook_handle.remove()
            self.steering_vector = None

    def compute_style_similarity(self, generated, ground_truth):
        len_ratio = min(len(generated), len(ground_truth)) / max(len(generated), len(ground_truth), 1)
        gen_words = set(generated.lower().split())
        gt_words = set(ground_truth.lower().split())
        if len(gen_words) == 0 or len(gt_words) == 0:
            word_overlap = 0
        else:
            word_overlap = len(gen_words & gt_words) / len(gen_words | gt_words)
        gen_punct = sum(1 for c in generated if c in "!?.,;:")
        gt_punct = sum(1 for c in ground_truth if c in "!?.,;:")
        punct_sim = 1 - min(abs(gen_punct - gt_punct) / max(gt_punct, 1), 1)
        score = 0.3 * len_ratio + 0.5 * word_overlap + 0.2 * punct_sim
        return score

    def evaluate_weights(self, weights, profile, train_turns):
        scores = []
        for turn in train_turns:
            prompt = build_prompt(turn, profile)
            response = self.generate_with_steering(prompt, weights)
            score = self.compute_style_similarity(response, turn["assistant"])
            scores.append(score)
        return sum(scores) / len(scores)


def optimize_persona(persona_id, optimizer, sigma=2.5, max_generations=15,
                    population_size=8, output_dir="optimization_results_sigma2.5"):

    print(f"\n{'='*60}")
    print(f"Optimizing: {persona_id} (σ={sigma})")
    print(f"{'='*60}")

    profile, train_turns = load_persona_data(persona_id)

    print(f"  Speaker role: {profile['speaker_role']}")
    print(f"  Train turns: {len(train_turns)}")

    es = cma.CMAEvolutionStrategy(
        x0=[0.0] * 5,
        sigma0=sigma,
        inopts={
            'popsize': population_size,
            'maxiter': max_generations,
            'verb_disp': 1,
            'verb_log': 0
        }
    )

    best_weights = None
    best_fitness = -float('inf')
    history = []

    start_time = time.time()
    generation = 0

    while not es.stop():
        generation += 1
        solutions = es.ask()

        fitnesses = []
        for weights in solutions:
            fitness = optimizer.evaluate_weights(weights, profile, train_turns)
            fitnesses.append(-fitness)

        es.tell(solutions, fitnesses)

        gen_best_idx = np.argmin(fitnesses)
        gen_best_fitness = -fitnesses[gen_best_idx]
        gen_best_weights = solutions[gen_best_idx]

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_weights = gen_best_weights.copy()

        l2_norm = np.sqrt(sum(w**2 for w in gen_best_weights))

        history.append({
            "generation": generation,
            "best_fitness": gen_best_fitness,
            "l2_norm": l2_norm
        })

        print(f"  Gen {generation}: fitness={gen_best_fitness:.4f}, L2={l2_norm:.2f}")

    elapsed_time = time.time() - start_time
    l2_norm = np.sqrt(sum(w**2 for w in best_weights))

    print(f"\n  ✅ Best fitness: {best_fitness:.4f}, L2: {l2_norm:.2f}")

    result = {
        "persona_id": persona_id,
        "speaker_role": profile["speaker_role"],
        "relationship": profile["relationship"],
        "best_weights": {f"R{i+1}": float(w) for i, w in enumerate(best_weights)},
        "best_fitness": float(best_fitness),
        "l2_norm": float(l2_norm),
        "settings": {"sigma": sigma, "alpha": optimizer.alpha},
        "elapsed_time_seconds": elapsed_time,
        "history": history
    }

    persona_output_dir = Path(output_dir) / persona_id
    persona_output_dir.mkdir(parents=True, exist_ok=True)

    with open(persona_output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    with open(persona_output_dir / "best_weights.json", "w") as f:
        json.dump({f"R{i+1}": float(w) for i, w in enumerate(best_weights)}, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--trait_dir", default="data/steering_vectors_v2")
    parser.add_argument("--output_dir", default="optimization_results_sigma2.5")
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--personas", nargs="+", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    print("Loading trait vectors...")
    trait_vectors = load_trait_vectors(args.trait_dir)

    optimizer = SteeringOptimizer(
        model=model,
        tokenizer=tokenizer,
        trait_vectors=trait_vectors,
        alpha=args.alpha
    )

    if args.personas:
        target_personas = args.personas
    else:
        target_personas = HIGH_L2_PERSONAS

    print(f"Target personas: {len(target_personas)}")
    print(f"Settings: σ={args.sigma}")

    results = []

    for i, persona_id in enumerate(target_personas):
        print(f"\n[{i+1}/{len(target_personas)}] {persona_id}")

        try:
            result = optimize_persona(
                persona_id=persona_id,
                optimizer=optimizer,
                sigma=args.sigma,
                output_dir=args.output_dir
            )
            results.append(result)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({"persona_id": persona_id, "error": str(e)})

    with open(output_dir / "optimization_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # サマリー出力
    successful = [r for r in results if "l2_norm" in r]
    if successful:
        avg_l2 = np.mean([r["l2_norm"] for r in successful])
        avg_fitness = np.mean([r["best_fitness"] for r in successful])
        print(f"\n{'='*60}")
        print(f"Summary: {len(successful)} personas optimized")
        print(f"Average L2: {avg_l2:.2f}")
        print(f"Average Fitness: {avg_fitness:.4f}")

    print(f"\n✅ 完了: {output_dir}")


if __name__ == "__main__":
    main()
