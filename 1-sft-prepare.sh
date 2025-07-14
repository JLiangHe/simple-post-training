conda activate llm_base

cd /gpfs/radev/home/jh3439/project/lm_sft

python src.hf_downloader.py --folder /gpfs/radev/project/zhuoran_yang/shared/datasets --type dataset teknium/OpenHermes-2.5
python src.hf_downloader.py --folder /gpfs/radev/project/zhuoran_yang/shared/datasets --type dataset SoftAge-AI/multi-turn_dataset

python -m src.data_processors.openhermes2_5
python -m src.data_processors.softageai
python -m src.data_processors.aggregate_and_split

python -m src.model_processor