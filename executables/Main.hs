{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}

module Main where

import HLearning.NeuralNetwork
import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra.Static

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
  print "hello world"

--  mapM putStrLn $ fmap show training

--readTrainingData :: IO [(Integer, [Integer])]
readTrainingData = readData "/Users/mikael/Documents/boulotmik/projects/hlearning/data/optdigits.tra"
readTestData     = readData "/Users/mikael/Documents/boulotmik/projects/hlearning/data/optdigits.tes"

--readCourseraTheta1 =

readCsvFile f =
  let file = readFile f
  in fmap parseCsvFile file

  -- returns label, feature vector
parseCsvLine :: String -> [Int]
parseCsvLine s = read ("[" ++ s ++ "]") :: [Int]

parseCsvFile :: String -> [[Int]]
parseCsvFile s = parseCsvLine <$> lines s

readData f =
  let file = readFile f
  in fmap parseFile file
