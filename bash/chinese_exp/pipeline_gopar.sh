data_dir=../../data/hsk_train
src_file=$data_dir/src.txt
tgt_file=$data_dir/tgt.txt
src_file_char=$data_dir/src.txt.char
tgt_file_char=$data_dir/tgt.txt.char
para_file=$data_dir/para_tgt2src.txt
m2_file=$data_dir/m2_reversed.txt
gopar_path=../../model/gopar/biaffine-dep-electra-zh-gopar
vanilla_parser_path=$gopar_path/off-the-shelf-model

# apply char
if [ ! -f $src_file".char" ]; then
  python ../../utils/segment_bert.py <$src_file >$src_file_char
  python ../../utils/segment_bert.py <$tgt_file >$tgt_file_char
fi

# # train off-the-shelf-parser
# mkdir -p $gopar_path

# TRAIN_CONLL=../../data/codt/train.conll
# DEV_CONLL=../../data/codt/dev.conll
# TEST_CONLL=../../data/codt/test.conll
# ## if your conll_file is word-level, you need to convert it from word to char
# # python ../../utils/convert_chinese_treebank_from_word_to_char.py $TRAIN_CONLL $TRAIN_CONLL".char"
# # python ../../utils/convert_chinese_treebank_from_word_to_char.py $DEV_CONLL $DEV_CONLL".char"
# # python ../../utils/convert_chinese_treebank_from_word_to_char.py $TEST_CONLL $TEST_CONLL".char"
# # TRAIN_CONLL=../../data/codt/train.conll.char
# # DEV_CONLL=../../data/codt/dev.conll.char
# # TEST_CONLL=../../data/codt/test.conll.char

# CUDA_VISIBLE_DEVICES=0 python -m supar.cmds.biaffine_dep train -b -d 0 -c ../../src/src_gopar/configs/ctb7.biaffine.dep.electra.ini -p $vanilla_parser_path -f char --encoder bert --bert hfl/chinese-electra-180g-large-discriminator \
#        --train $TRAIN_CONLL \
#        --dev $DEV_CONLL \
#        --test $TEST_CONLL \
#        --seed 1 \
#        --punct

# Step 1. Parse the target-side sentences in parallel GEC data by an off-the-shelf parser 
## If you find this step cost too much time, you can split the large file to several small files and predict them on multiple GPUs, and merge the results.
python ../../src/src_gopar/parse.py $tgt_file_char $tgt_file.conll_predict $vanilla_parser_path

# Step 2. Extract edits by ChERRANT from target-side to source-side
## Note:para_file should be tgt2src
cherrant_path=./cherrant  # You need to first download ChERRANT from https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT
cd $cherrant_path
paste $tgt_file_char $src_file_char | awk '{print NR"\t"$p}' > $para_file
python parallel_to_m2.py -f $para_file -o $m2_file -g char --segmented
cd -
 
# Step 3. Project the target-side trees to source-side ones
python ../../src/src_gopar/convert_gec_data_to_parsing_data_chinese.py $tgt_file.conll_predict $m2_file $src_file.conll_converted_gopar

# Step 4. Train GOPar
mkdir -p $gopar_path
python -m torch.distributed.launch --nproc_per_node=8 --master_port=10000 \
       -m supar.cmds.biaffine_dep train -b -d 0,1,2,3,4,5,6,7 -c ../../src/src_gopar/configs/ctb7.biaffine.dep.electra.ini -p $gopar_path/model -f char --encoder bert --bert hfl/chinese-electra-180g-large-discriminator \
       --train $src_file.conll_converted_gopar \
       --dev ../../data/mucgec_dev/src.txt.conll_converted_gopar \
       --test ../../data/mucgec_dev/src.txt.conll_converted_gopar \
       --seed 1 \
       --punct 

# Step 5. Predict source-side trees for GEC training
CoNLL_SUFFIX=conll_predict_gopar
IN_FILE=../../data/hsk+lang8_train/src.txt.char
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
CUDA_VISIBLE_DEVICES=0 nohup python ../../src/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path/model &

IN_FILE=../../data/mucgec_dev/src.txt.char
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
CUDA_VISIBLE_DEVICES=0 nohup python ../../src/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path/model &

IN_FILE=../../data/mucgec_test/src.txt.char
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
CUDA_VISIBLE_DEVICES=0 nohup python ../../src/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path/model &
