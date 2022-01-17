#!/bin/bash

#!/bin/bash

for i in */*.mp3;
  do name=`echo "$i" | cut -d'.' -f1`
  echo "$name"
  
  ffmpeg -i "$i" -acodec pcm_u8 -ar 16000 "${name}.wav"
done
