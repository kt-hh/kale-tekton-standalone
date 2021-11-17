pip3 install /root/kale-tekton-standalone/backend/.
cd /root/kale-tekton-standalone/labextension
jlpm install
jlpm run build
jupyter labextension install .

jupyter lab --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser \
      --allow-root --port=7179 --LabApp.token='' --LabApp.password='' \
      --LabApp.allow_origin='*'
