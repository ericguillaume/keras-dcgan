rm -rf /tmp/logs/remote
rm -rf /tmp/remote
mkdir -p /tmp/logs
mkdir -p /tmp/remote
gcloud compute scp try-cli-1:/tmp/logs /tmp/logs/remote --recurse --compress
gcloud compute scp try-cli-1:/home/eric/keras-dcgan/generated/last_last.png /tmp/remote/generated_last_last.png --compress
gcloud compute scp try-cli-1:/home/eric/keras-dcgan/generated /tmp/remote/generated --recurse --compress
