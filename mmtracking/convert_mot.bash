for i in 15 16 17 20
do
    echo "formating MOT$i"
    python tools/convert_datasets/mot/mot2coco.py -i /sdb_data/drive/MOT/MOT$i/ -o /sdb_data/drive/MOT/MOT$i/annotations/ --convert-det --split-train
done