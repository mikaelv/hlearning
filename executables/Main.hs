{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}

module Main where

import HLearning.NeuralNetwork
import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra.Static


main = do
  (labels, features) <- readTrainingData
  theta1 <- LA.rand 25 65
  theta2 <- LA.rand 10 26
  let Just theta1Typed = create theta1
      Just theta2Typed = create theta2
      initModel = Theta theta1Typed theta2Typed
  print (trainNetwork labels features initModel)
--  mapM putStrLn $ fmap show training

--readTrainingData :: IO [(Integer, [Integer])]
readTrainingData =
  let file = readFile "/Users/mikael/Documents/boulotmik/projects/hlearning/data/optdigits.tra"
  in fmap parseFile file
