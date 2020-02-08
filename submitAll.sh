#for m in "bpr" "apr" "sasrec" "asasrec"
#for m in "asasrec" "apr" "sasrec" "bpr"
for m in "asasrec2"
do
	#for data in "ml-1m-sort" "ml-1m-sort-dup" "yelp-sort" "yelp-sort-dup" "brightkite-sort" "brightkite-sort-dup" "fsq11-sort" "fsq11-sort-dup" "Video" "Beauty" "pinterest-20"
	for data in "ml-1m-sort" "yelp-sort" "brightkite-sort" "fsq11-sort" "Video" "pinterest-20"
	#for data in "ml-1m-sort"
	do
		#for d in 8 16 32 64
		#for e in 0.3 0.5 0.7 0.9 1
		for d in 64
		do
			#for l in 0.1 0.3 0.5 0.7 0.9 5
			for l in 1 
			do
				#for e in 0.1 0.3 0.7 0.9
				for e in 0.5
				do
					for ep in 0.01 0.1
					do
						for ed in 0.01 0.1
						do
							#for ec in 0.01 0.1
							for ec in 0.01 0.1
							do
								qsub script.sh $m $data $d $l $e $ep $ed $ec
							done
						done
					done
				done
			done
		done
	done
done
