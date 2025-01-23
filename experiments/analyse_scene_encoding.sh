for epoch in 25 50 200; do # Network training epochs
  for conf in 25 50; do # low and normal conf threshold
    python3.8 experiments/analyse_scene_encoding_det.py -i results/scene_encodings_det_softmax/scene_encoding_det_softmax_epoch${epoch}_conf${conf}.json
  done
done

# ///

for epoch in 25 50 200; do # Network training epochs
  for conf in 25 50; do # low and normal conf threshold
    for alpha in 05 15 25; do
      python3.8 experiments/analyse_scene_encoding_nondet.py -i results/scene_encodings_nondet_k1/scene_encoding_nondet_epoch${epoch}_conf${conf}_alpha${alpha}_k1.json
    done  
  done
done