mkdir output
cd tree2ctx
lein uberjar
cd ..
mkdir deps
cd deps
git clone https://github.com/angeloskath/py-kmeans
cd ..
