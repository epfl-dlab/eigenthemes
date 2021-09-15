rm -rf results/aida_testb_le_titov.tsv
python -m jrk.el_main --test_data data/aida_testb_le_titov.json --mode eval --model_path models/model_run1_prunedcands
python -m jrk.el_main --test_data data/aida_testb_le_titov.json --mode eval --model_path models/model_run2_prunedcands
python -m jrk.el_main --test_data data/aida_testb_le_titov.json --mode eval --model_path models/model_run3_prunedcands
python -m jrk.el_main --test_data data/aida_testb_le_titov.json --mode eval --model_path models/model_run4_prunedcands
python -m jrk.el_main --test_data data/aida_testb_le_titov.json --mode eval --model_path models/model_run5_prunedcands
echo "TauMIL-ND Results: CoNLL-Test (Overall)"
bash test_avg.sh results/aida_testb_le_titov.tsv

rm -rf results/aida_testb_le_titov_easy.tsv
python -m jrk.el_main --test_data data/aida_testb_le_titov_easy.json --mode eval --model_path models/model_run1_prunedcands
python -m jrk.el_main --test_data data/aida_testb_le_titov_easy.json --mode eval --model_path models/model_run2_prunedcands
python -m jrk.el_main --test_data data/aida_testb_le_titov_easy.json --mode eval --model_path models/model_run3_prunedcands
python -m jrk.el_main --test_data data/aida_testb_le_titov_easy.json --mode eval --model_path models/model_run4_prunedcands
python -m jrk.el_main --test_data data/aida_testb_le_titov_easy.json --mode eval --model_path models/model_run5_prunedcands
echo "TauMIL-ND Results: CoNLL-Test (Easy)"
bash test_avg.sh results/aida_testb_le_titov_easy.tsv

rm -rf results/aida_testb_le_titov_hard.tsv
python -m jrk.el_main --test_data data/aida_testb_le_titov_hard.json --mode eval --model_path models/model_run1_prunedcands
python -m jrk.el_main --test_data data/aida_testb_le_titov_hard.json --mode eval --model_path models/model_run2_prunedcands
python -m jrk.el_main --test_data data/aida_testb_le_titov_hard.json --mode eval --model_path models/model_run3_prunedcands
python -m jrk.el_main --test_data data/aida_testb_le_titov_hard.json --mode eval --model_path models/model_run4_prunedcands
python -m jrk.el_main --test_data data/aida_testb_le_titov_hard.json --mode eval --model_path models/model_run5_prunedcands
echo "TauMIL-ND Results: CoNLL-Test (Hard)"
bash test_avg.sh results/aida_testb_le_titov_hard.tsv

rm -rf results/wikipedia_complete_le_titov.tsv
python -m jrk.el_main --test_data data/wikipedia_complete_le_titov.json --mode eval --model_path models/model_run1_prunedcands
python -m jrk.el_main --test_data data/wikipedia_complete_le_titov.json --mode eval --model_path models/model_run2_prunedcands
python -m jrk.el_main --test_data data/wikipedia_complete_le_titov.json --mode eval --model_path models/model_run3_prunedcands
python -m jrk.el_main --test_data data/wikipedia_complete_le_titov.json --mode eval --model_path models/model_run4_prunedcands
python -m jrk.el_main --test_data data/wikipedia_complete_le_titov.json --mode eval --model_path models/model_run5_prunedcands
echo "TauMIL-ND Results: WNED-Wiki"
bash test_avg.sh results/wikipedia_complete_le_titov.tsv

rm -rf results/clueweb_complete_le_titov.tsv
python -m jrk.el_main --test_data data/clueweb_complete_le_titov.json --mode eval --model_path models/model_run1_prunedcands
python -m jrk.el_main --test_data data/clueweb_complete_le_titov.json --mode eval --model_path models/model_run2_prunedcands
python -m jrk.el_main --test_data data/clueweb_complete_le_titov.json --mode eval --model_path models/model_run3_prunedcands
python -m jrk.el_main --test_data data/clueweb_complete_le_titov.json --mode eval --model_path models/model_run4_prunedcands
python -m jrk.el_main --test_data data/clueweb_complete_le_titov.json --mode eval --model_path models/model_run5_prunedcands
echo "TauMIL-ND Results: WNED-Clueweb"
bash test_avg.sh results/clueweb_complete_le_titov.tsv
