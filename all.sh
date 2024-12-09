python3 main.py gen cityscapes --n_buildings 15 --n_cityscapes 150 --map_size 100 --world_size 150
python3 main.py gen windflow --speed_x 2 --speed_y 3 --pre_time 30 --post_time 1 --map_size 150
python3 main.py gen drone --radius 50 --num_positions 100
python3 main.py viz wind --index 10 --export-all --map_size 100 --world_size 150
python3 main.py gen lidar
python3 main.py viz wind  --index 10 --export-all --export-dir data/demoviz --map_size 150 --world_size 150
python3 main.py viz wind --index 10 --export-all-transparent --export-dir data/transparent --map_size 150 --world_size 150
