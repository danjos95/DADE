dataDir="../../data"  # path/to/your/data
config="../../cfgs/l_yolox_tal_filp.py" # path/to/your/cfg
weights="../../ckpts/yolox_l.pth"  # path/to/your/checkpoint_path

scale=0.5

python3 yolox_det.py \
	--data-root "$dataDir/Argoverse-1.1/tracking" \
	--annot-path "$dataDir/Argoverse-HD/annotations/val.json" \
	--fps 30 \
	--weights $weights \
	--in_scale 0.5 \
	--no-mask \
	--out-dir "../data/online_resuklt/yolox_s50" \
	--overwrite \
	--config $config \
   &&
python3 streaming_eval.py \
	--data-root "$dataDir/Argoverse-1.1/tracking" \
	--annot-path "$dataDir/Argoverse-HD/annotations/val.json" \
	--fps 30 \
	--eta 0 \
	--result-dir "../data/online_resuklt/yolox_s50" \
	--out-dir "../data/online_resuklt/yolox_s50" \
	#--vis-dir "/data/online_resuklt/l_s50/vis" \
	#--vis-scale 0.5 \
	--overwrite \
