for epoch in 25 50 200; do # Network training epochs
  for conf in 25 50; do # low and normal conf threshold
    /usr/local/bin/python3.8 reason.py -f results/scene_encodings_det_softmax/scene_encoding_det_softmax_epoch${epoch}_conf${conf}.json \
      -q data/CLEVR_v1.0/questions/CLEVR_val_questions.json \
      -o results/reasoning_det/reasoning_det_softmax_epoch${epoch}_conf${conf}.txt \
      -t utils/theory.lp \
      --answer_mode single
  done
done
