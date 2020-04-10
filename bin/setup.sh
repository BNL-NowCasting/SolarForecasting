#!/bin/bash
# source:
# wget https://raw.githubusercontent.com/BNL-NowCasting/SolarForecasting/master/bin/setup.sh
# setup solar nowcasting system
#
function die() {
    echo "$@" 2>&1
    echo "usage: " 2>&1
    echo "cd to user writable install directory" 2>&1
    echo "if desired use symlinks to other file-systems" 2>&1
    exit 1
}
function confirm () {
    echo -n "$@ y[n]: "
    read ANS
    case "$ANS" in
        [yY]*) return 0;;
        *) return 1;;
    esac
}

[ $EUID -eq 0 ] && die "do not run as root"
[ $USER = "nowcast" ] || confirm "recommend running as user 'nowcast', do you want to continue as user '$USER'" || exit 1
#
# Install everything relative CWD
NOWCASTHOME=`pwd`
# make sure it is writable
[ -w "${NOWCASTHOME}" ] || die "${NOWCASTHOME} is not writable"
# create data dirs
: ${DATAROOT="${NOWCASTHOME}/data/"}
for S in bnl alb
do
    SITE="${DATAROOT}${S}"
    mkdir -p ${SITE}
    chmod 2775 ${SITE}
    mkdir -p ${SITE}/{cache,latest,images}
    mkdir -p ${SITE}/log
    mkdir -p ${SITE}/{tmp,masks,stitch,feature,forecast,GHI}
done
# there may be more dirs to set up, but some may also be created within the python
# scripts that need them.
#
# Install anaconda version of python 3.6 in /opt/anaconda3, if it does not exist
: ${CONDADIR="${NOWCASTHOME}/anaconda3"}
: ${CONDA:="${CONDADIR}/bin/conda"}
: ${DOWNLOAD:="${NOWCASTHOME}/software"}
mkdir -p $DOWNLOAD
if [ ! -x "${CONDA}" ]
then
    ( cd $DOWNLOAD
      wget -nv https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh 
    )
    bash ${DOWNLOAD}/Anaconda3-2020.02-Linux-x86_64.sh -b -p ${CONDADIR}
    . ${CONDADIR}/bin/activate
    # update to latest version of conda
    conda update -n base -c defaults -y conda
    # the latest conda installs with python 3.7 by default in "base" environment,
    # but allows other versions of python to be installed as separate envs.
    # officially only python 2.7, 3.6 and 3.7 are support, but it seemed to create
    # a python 3.5 env. with this:
    conda create -y -n py35 python=3.5 anaconda
    conda activate py35
    conda init # sets env. in .bashrc
    # should log out or start new shell here to refresh env.
    . $NOWCASTHOME/.bashrc
    conda activate py35
    # use:
    # conda info --envs
    # to verify active env.
    conda install -y ephem
    # pyFFTW doesn't exist at anaconda, try pip instead.
    # upgrade pip
    pip install --upgrade pip
    pip install pyFFTW
fi

( cd $DOWNLOAD
    # get alpha release in dev branch
    wget -nv https://github.com/BNL-NowCasting/SolarForecasting/archive/v1.0.1-alpha.tar.gz
)
tar xzf ${DOWNLOAD}/v1.0.1-alpha.tar.gz
