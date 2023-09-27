data=rank_car
mkdir ../output/$data.results
for seed in {1..5}
do
  mkdir ../output/$data.results/seed.$seed
  for depth in {2..4..2}
  do
    mkdir ../output/$data.results/seed.$seed/depth.$depth
    for rf in {1..100..2}
    do
      python fun_with_trees.py --data=$data --target=class --mode=$rf --depth=$depth --seed=$seed
    done 
    mv ../output/rank_car ../output/$data.results/seed.$seed/depth.$depth/
  done
done
