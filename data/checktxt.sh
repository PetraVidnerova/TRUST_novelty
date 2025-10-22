for DIR in arxiv cell
do
    for F in ${DIR}/TXT/*.txt
    do
	LINES=`cat $F | wc --lines`
	if [ "$LINES" -gt 100 ]; then
	    echo -n ". "
	else
	    echo -n "$F "
	fi
    done
done
