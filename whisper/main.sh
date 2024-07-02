set -e

echo "WHISPER_MODEL: $WHISPER_MODEL"
echo "TRK_JOB_UID: $TRK_JOB_UID"
echo "TRK_INPUT_FILE: $TRK_INPUT_FILE"
echo "TRK_LANGUAGE: $TRK_LANGUAGE"

mkdir /task
cd /task
mkdir ./in
aws s3 sync s3://$S3_STORAGE_BUCKET/$TRK_JOB_UID/in/ ./in

mkdir ./out
cd ./out
whisper "../in/$TRK_INPUT_FILE" --output_format all \
    --model_dir /whisper_models \
    --model $WHISPER_MODEL \
    --language $TRK_LANGUAGE

# check if the output is empty
if [ ! "$(ls -A .)" ]; then
    echo "Error: No output generated"
    exit 1
fi

aws s3 sync ./ s3://$S3_STORAGE_BUCKET/$TRK_JOB_UID/out/

cd ..
touch done.🚜-flag
aws s3 cp done.🚜-flag s3://$S3_STORAGE_BUCKET/$TRK_JOB_UID/
