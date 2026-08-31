#!/bin/bash -e
#=====================================================================================

# indicate location of pytuq repo here
export PYTUQ_HOME=../../../../pytuq

# relative location of pytuq and uqpc apps
export PUQAPPS=$PYTUQ_HOME/apps
export UQPC=$PYTUQ_HOME/apps/uqpc

################################
##    0. Setup the problem    ##
################################

## Given mean and standard deviation of each normal random parameter
# inputs are diffusion_coeff, init_amplitude, init_variance
echo "400 100 " > param_margpc.txt
echo "1.0 0.25" >> param_margpc.txt
echo "0.01 0.0025" >> param_margpc.txt
# Hermite-Gaussian PC
PC_TYPE=HG
INPC_ORDER=1
# Creates input PC coefficient file pcf.txt (will have lots of zeros since we assume independent inputs)
${PUQAPPS}/pc_prep.py -f marg -i param_margpc.txt -p ${INPC_ORDER}

# Number of samples requested
NTRN=99 # Training
NTST=20  # Testing

# Extract dimensionality d (i.e. number of input parameters)
DIM=`awk 'NR==1{print NF}' pcf.txt`

# Output PC order
OUTPC_ORDER=3 

###############################
##  1. Prepare the inputs    ##
###############################

# Prepare inputs for the black-box model (use input PC to generate input samples for the model)
# This creates files ptrain.txt, ptest.txt (train/test parameter inputs), qtrain.txt, qtest.txt (corresponding train/test stochastic PC inputs)
${PUQAPPS}/pc_sam.py -f pcf.txt -t ${PC_TYPE} -n $NTRN
mv psam.txt ptrain.txt; mv qsam.txt qtrain.txt
${PUQAPPS}/pc_sam.py -f pcf.txt -t ${PC_TYPE} -n $NTST
mv psam.txt ptest.txt; mv qsam.txt qtest.txt

# create pnames.txt and outnames.txt with input parameter names and output QoI names
echo "diffusion_coef" > pnames.txt
echo "init_amplitude" >> pnames.txt
echo "init_variance" >> pnames.txt
echo "max_temp" > outnames.txt
echo "mean_temp" >> outnames.txt
echo "std_temp" >> outnames.txt
echo "cell_temp" >> outnames.txt

################################
## 2. Run the black-box model ##
################################

# Run the black-box model, both training and testing
# use parallel to launch multiple jobs at a time, each using 1 MPI rank
# ptrain.txt is N x d input matrix, each row is a parameter vector of size d
# ytrain.txt is N x o output matrix, each row is a output vector of size o 
# Similar for testing
parallel --jobs 4 --keep-order --colsep ' ' './main3d.gnu.MPI.ex inputs diffusion_coeff={1} init_amplitude={2} init_variance={3} datalog=datalog_train{#}.txt > stdoutlog_train{#}.txt 2>&1 ; tail -1 datalog_train{#}.txt' :::: ptrain.txt > ytrain.txt
parallel --jobs 4 --keep-order --colsep ' ' './main3d.gnu.MPI.ex inputs diffusion_coeff={1} init_amplitude={2} init_variance={3} datalog=datalog_test{#}.txt > stdoutlog_test{#}.txt 2>&1 ; tail -1 datalog_test{#}.txt' :::: ptest.txt > ytest.txt

################################
# 2. Run the black-box model (alternative using mpiexec)
################################

## Initialize job index
#job_index=0
#
## Read each line from ptrain.txt
#while IFS=' ' read -r diffusion_coeff init_amplitude init_variance; do
#
#    # Prepare the datalog and stdout log filenames
#    datalog_filename="datalog_train${job_index}.txt"
#    stdoutlog_filename="stdoutlog_train${job_index}.txt"
#
#    # Run the job with mpiexec
#    mpiexec -n 4 ./main3d.gnu.MPI.ex inputs diffusion_coeff=$diffusion_coeff init_amplitude=$init_amplitude init_variance=$init_variance datalog=$datalog_filename </dev/null > $stdoutlog_filename 2>&1
#
#    # If successful, output the last line of the datalog
#    if [ $? -eq 0 ]; then
#        tail -1 "$datalog_filename" >> ytrain.txt
#    fi
#    job_index=$((job_index+1))
#done < ptrain.txt
#
## Initialize job index
#job_index=0
#
## Read each line from ptest.txt
#while IFS=' ' read -r diffusion_coeff init_amplitude init_variance; do
#
#    # Prepare the datalog and stdout log filenames
#    datalog_filename="datalog_test${job_index}.txt"
#    stdoutlog_filename="stdoutlog_test${job_index}.txt"
#
#    # Run the job with mpiexec
#    mpiexec -n 4 ./main3d.gnu.MPI.ex inputs diffusion_coeff=$diffusion_coeff init_amplitude=$init_amplitude init_variance=$init_variance datalog=$datalog_filename </dev/null > $stdoutlog_filename 2>&1
#
#    # If successful, output the last line of the datalog
#    if [ $? -eq 0 ]; then
#        tail -1 "$datalog_filename" >> ytest.txt
#    fi
#    job_index=$((job_index+1))
#done < ptest.txt

##############################
#  3. Build PC surrogate    ##
##############################

# Build surrogate for each output (in other words, build output PC)
# This creates files results.pk (Python pickle file encapsulating the results)
${UQPC}/uq_pc.py -r offline -c pcf.txt -x ${PC_TYPE} -d $DIM -o ${INPC_ORDER} -m anl -s rand -n $NTRN -v $NTST -t ${OUTPC_ORDER}

###################################
##  4. Visualize the i/o data    ##
###################################

awk '{print "Training"}' ytrain.txt > labels.txt
awk '{print "Testing"}' ytest.txt >> labels.txt
cat ytrain.txt ytest.txt > yall.txt
cat ptrain.txt ptest.txt > pall.txt

${PUQAPPS}/plot_xx.py -x pall.txt -l labels.txt
${PUQAPPS}/plot_pcoord.py -x pall.txt -y yall.txt  -l labels.txt

${PUQAPPS}/plot_yx.py -x ptrain.txt -y ytrain.txt -c 3 -r 1
${PUQAPPS}/plot_yxx.py -x ptrain.txt -y ytrain.txt
${PUQAPPS}/plot_pdfs.py -p ptrain.txt; cp pdf_tri.png pdf_tri_inputs.png
${PUQAPPS}/plot_pdfs.py -p ytrain.txt; cp pdf_tri.png pdf_tri_outputs.png
${PUQAPPS}/plot_ens.py -y ytrain.txt

# A lot of .png files are created for visualizing the input/output data

################################
## 5. Postprocess the results ##
################################

# Plot model vs PC for each output
${UQPC}/plot.py dm training testing
# Plot model vs PC for each sample
${UQPC}/plot.py fit training testing

# Plot output pdfs
${UQPC}/plot.py pdf
# Plot output pdfs in joyplot format
${UQPC}/plot.py joy

# Plot main Sobol sensitivities (barplot)
${UQPC}/plot.py sens main
# Plot joint Sobol sensitivities (circular plot)
${UQPC}/plot.py jsens
# Plot matrix of Sobol sensitivities (rectangular plot)
${UQPC}/plot.py sensmat main

# Plot 1d slices of the PC surrogate
${UQPC}/plot.py 1d training testing
# Plot 2d slices of the PC surrogate
${UQPC}/plot.py 2d 

# This creates .png files for visualizing PC surrogate results and sensitivity analysis
