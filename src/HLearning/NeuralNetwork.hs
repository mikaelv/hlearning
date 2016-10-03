{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}

module HLearning.NeuralNetwork where

import GHC.TypeLits
import Numeric.LinearAlgebra.Static
import qualified Numeric.LinearAlgebra as LA
import qualified Data.Vector.Storable as V
import HLearning.Util


trainNetwork :: [Int] -> [[Int]] -> (L 25 65, L 10 26)
trainNetwork labels features =
  let theta1Init = konst 1-- TODO random init
      theta2Init = konst 1
      Just y = (create . listToVector) labels :: Maybe (R 3823)
      Just x = (create . LA.fromRows . map listToVector) features :: Maybe (L 3823 64)
      initCost = cost x y theta1Init theta2Init
  in (theta1Init, theta2Init)

listToVector = LA.vector . map fromIntegral

-- returns label, feature vector
parseLine :: String -> (Int, [Int])
parseLine s =
  let ints = (reverse . read) ("[" ++ s ++ "]") :: [Int]
  in (head ints, reverse $ tail ints)

parseFile :: String -> ([Int], [[Int]])
parseFile s = unzip $ parseLine <$> lines s

-- x: nb of training examples * nb_of_features
-- y: label corresponding to training examples
-- theta1 : nb of rows in hidden layer * (nb_of_features +1)
-- theta2 : nb of rows in output layer * (nb_of_rows(theta1) +1)
cost :: L 3823 64 -> R 3823 -> L 25 65 -> L 10 26 -> Double
cost x y theta1 theta2 =
  let a1 = (addOnes . tr) x
      z2 = theta1 <> a1
      a2 = addOnes $ sigmoid z2
      z3 = theta2 <> a2
      a3 = sigmoid z3
      h = tr a3
      yb = toBinMatrix y
      jmat = yb * log h - yb * log (1 - h)
      m = fromIntegral $ size y
      -- TODO regularization
  in - (LA.sumElements . extract) jmat / m


backpropagate :: L 3823 10 -> Double -> L 25 65 -> L 10 26 -> L 25 3823 -> L 65 3823 -> L 26 3823 -> L 10 3823 -> Double
backpropagate yb lambda theta1 theta2 z2 a1 a2 a3 =
  let delta3 = a3 - tr yb
      delta2a = tr theta2 <> delta3
      delta2b = snd (splitRows delta2a :: (L 1 3823, L 25 3823)) -- TODO replace 3823 with m does not compile here !?
      delta2 = delta2b * sigmoidGradient z2
      d2 = delta3 <> tr a2
      d1 = delta2 <> tr a1
      m = 3823
      regulCoef = lambda / m
      gradTheta1 = (d1 / konst m) + (konst regulCoef * theta1)
  in 3

multScalar :: (KnownNat m, KnownNat n) => L m n -> Double -> L m n
multScalar m d = m * konst d

sigmoidGradient :: (KnownNat m, KnownNat n) => L m n -> L m n
sigmoidGradient m = g * (1-g)
  where g = sigmoid m


--   -- read this: https://www.schoolofhaskell.com/user/konn/prove-your-haskell-for-great-safety/dependent-types-in-haskell
sigmoid :: (KnownNat m, KnownNat n) => L m n -> L m n
sigmoid a = 1 / (1 + exp(-a))

addOnes :: (KnownNat m, KnownNat n) => L m n -> L (1+m) n
addOnes x = konst 1 === x
--costTheta =
