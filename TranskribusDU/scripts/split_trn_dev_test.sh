#!/bin/sh

col="$1"

#percentage of each
p_trn=80
p_dev=5
p_tst=15

if [ ! -d "$col" ]
then
  echo " USAGE: $0 collection_folder"
  exit 1
else
	echo "Splitting $col in $1_trn $1_dev $1_tst"
fi

echo "$p_trn% training   $p_tst% testing   $p_dev% dev"

pushd "$col" > /dev/null
coldir=`pwd`
popd > /dev/null

\ls "$coldir"/*.mpxml > lst_doc

shuf lst_doc > lst_doc_shuffled

n=`wc -l < lst_doc`
echo "$n documents"

n_tst=$(($p_tst*$n/100))
n_dev=$(($p_dev*$n/100))
n_trn=$(($n-$n_tst-$n_dev))
echo "$n_trn training   $n_dev dev   $n_tst testing"

head --lines=$n_trn lst_doc_shuffled > lst_trn

start=$(($n_trn+1))
end=$(($start+$n_dev))
tail -n +$start lst_doc_shuffled | head --lines=$n_dev > lst_dev

start=$(($end))
end=$(($start+$n_tst))
tail -n +$start lst_doc_shuffled  > lst_tst

wc lst_trn lst_dev lst_tst

cat lst_trn lst_dev lst_tst > /tmp/lst_selftest
diff lst_doc_shuffled /tmp/lst_selftest

splitdir="$coldir.split_${p_trn}_${p_dev}_${p_tst}"
mkdir $splitdir
mv lst_trn lst_dev lst_tst lst_doc lst_doc_shuffled /tmp/lst_selftest "$splitdir"
echo "$n documents" >"$splitdir"/MEMO.txt
echo "$n_trn training   $n_dev dev   $n_tst testing" >> "$splitdir"/MEMO.txt

mkdir $splitdir/TRN
mkdir $splitdir/TRN/col
pushd $splitdir/TRN/col
for i in `cat ../../lst_trn`
do
	#ln -s $coldir/$i .
	ln -s $i .
done
popd

mkdir $splitdir/DEV
mkdir $splitdir/DEV/col
pushd $splitdir/DEV/col
for i in `cat ../../lst_dev`
do
	ln -s $i .
done
popd

mkdir $splitdir/TST
mkdir $splitdir/TST/col
pushd $splitdir/TST/col
for i in `cat ../../lst_tst`
do
	ln -s $i .
done
popd

chmod ugo-w "$splitdir" "$splitdir"/*
chmod ugo-r "$splitdir"/TST



