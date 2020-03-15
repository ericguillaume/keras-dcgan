rm -rf /tmp/default_logs && rm -rf generated && mkdir -p generated && mkdir -p generated/model && python3 dcgan.py --mode train --batch_size 128
