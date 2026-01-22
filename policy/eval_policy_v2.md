Evaluation Policy v2

Criteria
1 Helpfulness
2 Correctness
3 Clarity
4 Specificity
5 Instruction following
6 Conciseness

Weights
Helpfulness 0.35, Correctness 0.20, Clarity 0.15, Specificity 0.15, Instruction 0.10, Conciseness 0.05

Decision
margin = total_B - total_A; tie_threshold=0.03
margin>threshold→B, margin<-threshold→A, else→TIE
Confidence = 0.50 + min(0.49, |margin|)

Persona fit (tie-breaker)
合成差が閾値未満のときのみ、与えられたR1〜R5の目標への適合を0〜1で内部採点し、差が大きい側を優先。
fitを理由に事実誤りを上書きしない。
