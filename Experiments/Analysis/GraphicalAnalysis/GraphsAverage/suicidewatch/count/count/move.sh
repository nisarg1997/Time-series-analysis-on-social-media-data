# mkdir ../GraphsAverage/addiction/count
# mv addiction/* ../GraphsAverage/addiction/count


#! /bin/bash

array=$( ls . ) 

for i in "${array}"
do
   :
   mkdir "../GraphsAverage/"$i"/count"
   cp $i"/*" "../GraphsAverage/"$i"/count"
   
done