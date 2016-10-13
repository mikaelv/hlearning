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
import Debug.Trace

data NNModel = Theta { theta1 :: L 25 65, theta2 :: L 10 26 } deriving Show

predict :: NNModel -> [Int] -> Int
predict nnModel features =
  let Just x = listToVector features :: Maybe (R 64)
      (h, a3, z3, a2, z2, a1) = feedForward nnModel (row x)
  in (V.maxIndex . unwrap . unrow) h

-- m: number of training examples
feedForward :: (KnownNat m) => NNModel -> L m 64 -> (L m 10, L 10 m, L 10 m, L 26 m, L 25 m, L 65 m)
feedForward (Theta theta1 theta2) x =
  let a1 = (addOnes . tr) x
      z2 = theta1 <> a1
      a2 = addOnes $ sigmoid z2
      z3 = theta2 <> a2
      a3 = sigmoid z3
      h = tr a3
  in (h, a3, z3, a2, z2, a1)


trainNetwork :: [Int] -> [[Int]] -> NNModel -> NNModel
trainNetwork labels features initModel =
  let Just y = listToVector labels :: Maybe (R 3823)
      Just x = listOfListToMatrix features :: Maybe (L 3823 64)
      lambda = 1 -- regularisation
      function = cost x y lambda
      tolerance = 1e-5
      --stop (Params prev) (Params cur) = abs (fst (function prev) - fst (function cur)) < tolerance
      stop (Params prev) (Params cur) = abs (fst (function cur)) < 0.1
      stopCond = StopWhen stop
      alpha = 0.01
      nnModel = model $ gradientDescent function stopCond alpha (Params initModel)
  in nnModel

listToVector :: KnownNat m => [Int] -> Maybe (R m)
listToVector lst = (create . LA.vector . map fromIntegral) lst :: KnownNat m1 => Maybe (R m1)

listOfListToMatrix :: (KnownNat m, KnownNat n) => [[Int]] -> Maybe (L m n)
listOfListToMatrix listOfList = (create . LA.fromRows . map (LA.vector . map fromIntegral) ) listOfList :: (KnownNat m1, KnownNat n1) => Maybe (L m1 n1)

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
cost x y lambda nnModel =
  let (h, a3, z3, a2, z2, a1) = feedForward nnModel x
      yb = toBinMatrix y
      jmat = yb * log h - yb * log (1 - h)
      m = fromIntegral $ size y
      -- TODO regularization
      j = - (LA.sumElements . extract) jmat / m
      (gradTheta1, gradTheta2) = backpropagate yb lambda nnModel z2 a1 a2 a3
  in (traceShowId j, Theta gradTheta1 gradTheta2)


backpropagate :: L 3823 10 -> Double -> NNModel -> L 25 3823 -> L 65 3823 -> L 26 3823 -> L 10 3823 -> (L 25 65, L 10 26)
backpropagate yb lambda (Theta theta1 theta2) z2 a1 a2 a3 =
  let delta3 = a3 - tr yb
      delta2a = tr theta2 <> delta3
      delta2b = snd (splitRows delta2a :: (L 1 3823, L 25 3823)) -- TODO replace 3823 with m does not compile here !?
      delta2 = delta2b * sigmoidGradient z2
      d2 = delta3 <> tr a2
      d1 = delta2 <> tr a1
      m = (fromIntegral . fst . size) yb
      gradTheta1NonReg  = (d1 / konst m)
      gradTheta2NonReg  = (d2 / konst m)
--      regulCoef = lambda / m
--      gradTheta1 = (d1 / konst m) + (konst regulCoef * theta1) -- TODO not for the first column
  in (gradTheta1NonReg, gradTheta2NonReg) -- TODO regularization



instance GradientDescent (NNModel -> (Double, NNModel)) where
  data Params (NNModel -> (Double, NNModel)) = Params { model :: NNModel }


  grad f (Params nnModel) =
    let (cost, gradTheta) = f nnModel
    in Params gradTheta

  -- Use numeric differentiation for taking the gradient.
  -- TODO compute for each element of the unrolled vector
  -- grad f (Params nnModel) =
  --   let (costA, _) = f nnModel
  --       (costB, _) = f (nnModel - konst epsilon)
  --   in Params $ (costA - costB) / epsilon
  --   where epsilon = 0.0001



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
