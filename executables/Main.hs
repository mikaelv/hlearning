{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}

module Main where

import HLearning.NeuralNetwork


main = do
  training <- readTrainingData
  mapM putStrLn $ fmap show training

--readTrainingData :: IO [(Integer, [Integer])]
readTrainingData =
  let file = readFile "/Users/mikael/Documents/boulotmik/projects/hlearning/data/optdigits.tra"
  in fmap parseFile file
