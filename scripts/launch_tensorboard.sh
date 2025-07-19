# Tensorboard needs to run on the MAIN node.
tensorboard \
  --logdir ../logs \
  --host 0.0.0.0 \
  --port 6006

# To view the tensorboard dashboard locally:
#
# 1. Start the tensorboard process on the main node.
# 2. Set up an SSH tunnel from your local machine to the MAIN node that
#    forwards the above port to a local machine.
#
#    ssh -L 6006:localhost:6006 user@MAIN_IP
#
# 3. On your local machine pull up the dashboard in your browser at
#    the following URL:
#
#    http://localhost:6006
