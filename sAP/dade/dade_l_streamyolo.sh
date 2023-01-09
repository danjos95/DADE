dataDir="../../data"  # path/to/your/data
config="../../cfgs/l_s50_onex_dade_tal_filp.py" # path/to/your/cfg
weights="../../ckpts/l_s50_one_x.pth"  # path/to/your/checkpoint_path

scale=0.5

python dade_det.py \
	--data-root "$dataDir/Argoverse-1.1/tracking" \
	--annot-path "$dataDir/Argoverse-HD/annotations/val.json" \
	--fps 30 \
	--weights $weights \
	--in_scale 0.5 \
	--no-mask \
	--out-dir "../data/online_resuklt/dade_l_s50" \
	--overwrite \
	--config $config \
   &&
python streaming_eval.py \
	--data-root "$dataDir/Argoverse-1.1/tracking" \
	--annot-path "$dataDir/Argoverse-HD/annotations/val.json" \
	--fps 30 \
	--eta 0 \
	--result-dir "../data/online_resuklt/dade_l_s50" \
	--out-dir "../data/online_resuklt/dade_l_s50" \
	--vis-dir "../data/online_resuklt/dade_l_s50/vis" \
	--vis-scale 0.5 \
	--overwrite \
