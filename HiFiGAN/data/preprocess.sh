oldIFS=$IFS
IFS=$'\n'

for i in `ls ./*.mp3`
do
    filepres=$(basename $i .mp3)
    #echo $filepres
    ffmpeg -i "$i" -sample_fmt s16 -ar 22050 -ac 1 ../zxc/${filepres}.wav;  # 用ffmpeg将flac格式的后缀加上.wav后缀
done

IFS=$oldIFS

cd ..
python split_audio.py