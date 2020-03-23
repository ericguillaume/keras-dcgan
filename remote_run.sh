gcloud compute scp /Users/eric/dev/data/anime-faces-processed_5k.tar.gz try-cli-1:/tmp/anime-faces-processed_5k.tar.gz --compress
gcloud compute ssh try-cli-1 --zone=europe-west2-c < remote_run_impl.sh
