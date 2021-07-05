wget https://users.dcc.uchile.cl/~jperez/beto/cased_2M/pytorch_weights.tar.gz
wget https://users.dcc.uchile.cl/~jperez/beto/cased_2M/vocab.txt
wget https://users.dcc.uchile.cl/~jperez/beto/cased_2M/config.json
tar -xzvf pytorch_weights.tar.gz
mv config.json pytorch/.
mv vocab.txt pytorch/.