for DIR in cell arxiv 
do
    I=0
    for F in ${DIR}/PDF/*.pdf
    do
	B=`basename $F .pdf`
	N="${DIR}/TXT/$B.txt"
	if [ -f $N ]; then
	    echo "$N exists"
	else
	    I=`expr $I + 1`
	    echo $I
	    echo -n $B ...
	    pdf2txt $F > $N &
	    if [ $I -gt 5 ]; then
		wait -n
		echo "converted"
	    fi
	fi
    done
done
