# Define available curve libraries with an index and their supported features
# Format: index:curve:features
set(ICICLE_CURVES
  1:bn254:NTT,MSM,G2,ECNTT,POSEIDON,POSEIDON2,SUMCHECK,FRI,PAIRING
  2:bls12_381:NTT,MSM,G2,ECNTT,POSEIDON,POSEIDON2,SUMCHECK,FRI,PAIRING
)