python3 main.py gen cityscapes
python3 main.py gen windflow
python3 main.py gen drone --radius 30,40,50
python3 main.py viz wind --index 10 --export-all
python3 main.py gen lidar
python3 main.py viz wind  --index 10 --export-all --export-dir data/demoviz --map_size 150 --world_size 150
python3 main.py viz wind --index 10 --export-all-transparent --export-dir data/transparent --map_size 150 --world_size 150
