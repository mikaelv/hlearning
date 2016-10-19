{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE Rank2Types #-}


module Main where

import HLearning.NeuralNetwork
import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra.Static
import Debug.Trace
import GHC.TypeLits

main = mainCoursera

mainNIST = do
  (labels, features) <- readTrainingData
  theta1 <- LA.rand 25 65
  theta2 <- LA.rand 10 26
  let Just theta1Typed = create theta1
      Just theta2Typed = create theta2
      Just y = listToVector labels :: Maybe (R 3823)
      Just x = listOfListToMatrix features :: Maybe (L 3823 64)
      initModel = Theta theta1Typed theta2Typed
      trainedModel = trainNetwork y x initModel
  print "prediction for 1st example"
  (tstLabels, tstFeatures) <- readTestData
  print "hello world"
  -- print $ predict trainedModel (features !! 0)
  -- print $ predict trainedModel (features !! 1)
  -- print $ predict trainedModel (features !! 2)
  -- print $ predict trainedModel (features !! 3)
  -- print $ predict trainedModel (features !! 4)
  -- print $ predict trainedModel (features !! 5)



mainCoursera = do
  y <- readCsvFile "/Users/mikael/Documents/boulotmik/projects/coursera/machine-learning-ex4/ex4/ex4_y.csv"
  x <- readCsvFile "/Users/mikael/Documents/boulotmik/projects/coursera/machine-learning-ex4/ex4/ex4_X.csv"
  theta1 <- readCsvFile "/Users/mikael/Documents/boulotmik/projects/coursera/machine-learning-ex4/ex4/ex4_theta1.csv"
  theta2 <- readCsvFile "/Users/mikael/Documents/boulotmik/projects/coursera/machine-learning-ex4/ex4/ex4_theta2.csv"
  let Just y' = listOfListToMatrix $ fixOctaveLabels y :: Maybe (L 5000 1)
      y'' = (unrow . tr) $ traceMatrix "y''" y'
      Just x' = listOfListToMatrix x :: Maybe (L 5000 400)
      Just theta1' = listOfListToMatrix theta1 :: Maybe (L 25 401)
      Just theta2' = listOfListToMatrix theta2 :: Maybe (L 10 26)
      (j, grad) = traceShow "cost" $ cost x' y'' 1 (Theta (traceMatrix "theta1" theta1') (traceMatrix "theta2" theta2'))
  print $ "cost at parameters loaded from ex4_theta?.csv (should be 0.287629): " ++ (show j)

fixOctaveLabels :: [[Double]] -> [[Double]]
fixOctaveLabels = fmap $ fmap (\x -> x -1) 

 
--readTrainingData :: IO [(Integer, [Integer])]
readTrainingData = readData "/Users/mikael/Documents/boulotmik/projects/hlearning/data/optdigits.tra"
readTestData     = readData "/Users/mikael/Documents/boulotmik/projects/hlearning/data/optdigits.tes"

--readCourseraTheta1 =

readCsvFile f =
  let file = readFile f
  in fmap parseCsvFile file

  -- returns label, feature vector
parseCsvLine :: String -> [Double]
parseCsvLine s = read ("[" ++ s ++ "]") :: [Double]

parseCsvFile :: String -> [[Double]]
parseCsvFile s = parseCsvLine <$> lines s

readData f =
  let file = readFile f
  in fmap parseFile file
