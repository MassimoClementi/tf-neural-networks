echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating the virtual environment..."
source venv/bin/activate

echo "Installing the required packages..."
pip3 install -r requirements.txt

#echo "Configuring the jupiter kernel"
#venv/bin/python3 -m ipykernel install --prefix=./jupiter/ --name='venv-jupiter-kernel'

#echo "Showing installed kernels..."
#jupyter kernelspec list
