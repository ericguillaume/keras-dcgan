gcloud compute instances create try-cli-1 --zone=us-central1-b --image-project=deeplearning-platform-release --image-family=tf-2-1-cu100 --maintenance-policy=TERMINATE --accelerator="type=nvidia-tesla-t4,count=1" --metadata="install-nvidia-driver=True" --preemptible
# --image-family = tf2-latest-gpu | tf-2-1-cu100 | tf2-2-0-cu100
