count=0
while true; 
do 
	#if [[ $(net_segment train -c extensions/configs/config2.ini) = *Killed* ]]; then
	  net_segment train -c extensions/configs/config2.ini
	#fi
echo "$count"
count=$((count+1))
done
