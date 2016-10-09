{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module HLearning.NeuralNetwork where

import GHC.TypeLits
import Numeric.LinearAlgebra.Static
import qualified Numeric.LinearAlgebra as LA
import qualified Data.Vector.Storable as V
import HLearning.Util
import HLearning.GradientDescent

data NNModel = Theta { theta1 :: L 25 65, theta2 :: L 10 26 } deriving Show


trainNetwork :: [Int] -> [[Int]] -> NNModel -> NNModel
trainNetwork labels features initModel =
  let Just y = (create . listToVector) labels :: Maybe (R 3823)
      Just x = (create . LA.fromRows . map listToVector) features :: Maybe (L 3823 64)
      lambda = 1 -- regularisation
      function = cost x y lambda
      tolerance = 1e-9
      stop (Params prev) (Params cur) = abs (fst (function prev) - fst (function cur)) < tolerance
      stopCond = StopWhen stop
      alpha = 0.1
      nnModel = model $ gradientDescent function stopCond alpha (Params initModel)
  in nnModel

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
cost :: L 3823 64 -> R 3823 -> Double -> NNModel -> (Double, NNModel)
cost x y lambda (Theta theta1 theta2) =
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
      j = - (LA.sumElements . extract) jmat / m
      (gradTheta1, gradTheta2) = backpropagate yb lambda theta1 theta2 z2 a1 a2 a3
  in (j, Theta gradTheta1 gradTheta2)


backpropagate :: L 3823 10 -> Double -> L 25 65 -> L 10 26 -> L 25 3823 -> L 65 3823 -> L 26 3823 -> L 10 3823 -> (L 25 65, L 10 26)
backpropagate yb lambda theta1 theta2 z2 a1 a2 a3 =
  let delta3 = a3 - tr yb
      delta2a = tr theta2 <> delta3
      delta2b = snd (splitRows delta2a :: (L 1 3823, L 25 3823)) -- TODO replace 3823 with m does not compile here !?
      delta2 = delta2b * sigmoidGradient z2
      d2 = delta3 <> tr a2
      d1 = delta2 <> tr a1
      m = 3823
      gradTheta1NonReg  = (d1 / konst m)
      gradTheta2NonReg  = (d2 / konst m)
      regulCoef = lambda / m
      gradTheta1 = (d1 / konst m) + (konst regulCoef * theta1) -- TODO not for the first column
  in (gradTheta1NonReg, gradTheta2NonReg)



instance GradientDescent (NNModel -> (Double, NNModel)) where
  data Params (NNModel -> (Double, NNModel)) = Params { model :: NNModel }


  grad f (Params nnModel) =
    let (cost, gradTheta) = f nnModel
    in Params gradTheta


  paramMove scale (Params (Theta gradTheta1 gradTheta2)) (Params (Theta theta1 theta2)) =
    let newTheta1 = theta1 + gradTheta1 * konst scale
        newTheta2 = theta2 + gradTheta2 * konst scale
    --let newLocation = old + fromRational (toRational scale) * vec
    in Params (Theta newTheta1 newTheta2)
    --in VecArg newLocation


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
