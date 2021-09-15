# pull original LM-BFF code at a86087f to folder tmp
git clone git@github.com:princeton-nlp/LM-BFF.git
cd LM-BFF
git checkout a86087f
cd ..

# apply CLUES patch
patch -s -p0 < prompt.patch
mv LM-BFF/* .
rm -rf LM-BFF # clean up
