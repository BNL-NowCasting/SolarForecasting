#!/bin/bash
# better to do this in a python module that reads the dirs from the config file.
#
# make sure user nowcast exists
# also need group nowcast
getent group nowcast >/dev/null || sudo groupadd nowcast
getent passwd nowcast >/dev/null || sudo useradd -c "camera nologin" -s /sbin/nologin -g nowcast nowcast
# Install everything relative to nowcast's home
NOWCASTHOME=`getent passwd nowcast|cut -d: -f 6`
# create data dirs
: ${DATAROOT="${NOWCASTHOME}/data"}
declare -a SITES
SITES=( bnl alb )
for S in ${SITES[@]}; do
    SITE="${DATAROOT}${S}"
    sudo mkdir -p ${SITE}
    sudo chown nowcast:nowcast ${SITE}
done
# from here on in everything shoud run as user nowcast, best to split of into
# separate script
sudo -u nowcast bash <<EOF
for S in ${SITES[@]}; do
    SITE="${DATAROOT}${S}"
    chmod 2775 ${SITE}
    mkdir -p ${SITE}/{cache,latest,images}
    mkdir -p ${SITE}/log
    mkdir -p ${SITE}/{tmp,masks,stitch,feature,forecast,GHI}
done
# there may be more dirs to set up, but some may also be created within the python
# scripts that need them.
#
# Create a release of the latest commit in the HistoricProcessing-dev branch

#
# Install anaconda version of python 3.6 in /opt/anaconda3, if it doesn't exist
: ${CONDADIR="${NOWCASTHOME/anaconda3"}
: ${CONDA:="${CONDADIR}/bin/conda"}
: ${DOWNLOAD:="${NOWCASTHOME}/software"}
if [ ! -x "${CONDA}" ]
then
    mkdir -p $DOWNLOAD
    cd $DOWNLOAD
    wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
    bash ./Anaconda3-2020.02-Linux-x86_64.sh -b -p ${CONDADIR}
    . ${CONDADIR/bin/activate}
    # update to latest version of conda
    conda update -n base -c defaults -y conda
    # the latest conda installs with python 3.7 by default in "base" environment,
    # but allows other versions of python to be installed as separate envs.
    # officially only python 2.7, 3.6 and 3.7 are support, but it seemed to create
    # a python 3.5 env. with this:
    conda create -n py35 python=3.5 anaconda
    conda activate py35
    conda init # sets env. in .bashrc
    # should log out or start new shell here to refresh env.
    . $NOWCASTHOME/.bashrc
    conda activate py35
    # use:
    # conda info --envs
    # to verify active env.
    conda install ephem
    # pyFFTW doesn't exist at anaconda, try pip instead.
    # upgrade pip
    pip install --upgrade pip
    pip install pyFFTW
fi

EOF
