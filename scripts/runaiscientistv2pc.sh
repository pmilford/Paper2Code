# export OPENAI_API_KEY=""

$Env:GPT_VERSION="o4-mini"

$Env:PAPER_NAME="aiscientistv2"
$Env:PDF_PATH="../examples/aiscientistv2.pdf" # .pdf
$Env:PDF_JSON_PATH="../examples/aiscientistv2.json" # .json
$Env:PDF_JSON_CLEANED_PATH="../examples/aiscientistv2b_cleaned.json" # _cleaned.json
$Env:OUTPUT_DIR="../outputs/aiscientistv2b"
$Env:OUTPUT_REPO_DIR="../outputs/aiscientistv2b_repo"

mkdir -p $Env:OUTPUT_DIR
mkdir -p $Env:OUTPUT_REPO_DIR

echo $Env:PAPER_NAME

echo "------- Preprocess -------"

python ../codes/0_pdf_process.py `
    --input_json_path $Env:PDF_JSON_PATH `
    --output_json_path $Env:PDF_JSON_CLEANED_PATH `


echo "------- PaperCoder -------"

python ../codes/1_planning.py `
    --paper_name $Env:PAPER_NAME `
    --gpt_version $Env:GPT_VERSION `
    --pdf_json_path $Env:PDF_JSON_CLEANED_PATH `
    --output_dir $Env:OUTPUT_DIR


python ../codes/1.1_extract_config.py `
    --paper_name $Env:PAPER_NAME `
    --output_dir $Env:OUTPUT_DIR

cp -rp $Env:OUTPUT_DIR/planning_config.yaml $Env:OUTPUT_REPO_DIR/config.yaml

python ../codes/2_analyzing.py `
    --paper_name $Env:PAPER_NAME `
    --gpt_version $Env:GPT_VERSION `
    --pdf_json_path $Env:PDF_JSON_CLEANED_PATH `
    --output_dir $Env:OUTPUT_DIR

python ../codes/3_coding.py  `
    --paper_name $Env:PAPER_NAME `
    --gpt_version $Env:GPT_VERSION `
    --pdf_json_path $Env:PDF_JSON_CLEANED_PATH `
    --output_dir $Env:OUTPUT_DIR `
    --output_repo_dir $Env:OUTPUT_REPO_DIR `
