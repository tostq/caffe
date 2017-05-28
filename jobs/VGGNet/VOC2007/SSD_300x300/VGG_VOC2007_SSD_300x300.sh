cd /home/lee/caffe
./build/tools/caffe train \
--solver="models/VGGNet/VOC2007/SSD_300x300/solver.prototxt" \
--weights="models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 0 2>&1 | tee jobs/VGGNet/VOC2007/SSD_300x300/VGG_VOC2007_SSD_300x300.log
