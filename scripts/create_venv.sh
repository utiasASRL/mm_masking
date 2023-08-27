# Assumes that ROOTDIR is set and pointing to mm_masking root directory
cd $ROOTDIR
python3 -m venv venv
source venv/bin/activate
pip install -r $ROOTDIR/requirements.txt
deactivate

cd $ROOTDIR