{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE Rank2Types #-}

module HLearning.NeuralNetwork where

import GHC.TypeLits
import Numeric.LinearAlgebra.Static
import qualified Numeric.LinearAlgebra as LA
import qualified Data.Vector.Storable as V
import HLearning.Util
import HLearning.GradientDescent
import Debug.Trace

-- n1: nb of features + 1 (bias layer)
data NNModel n1 = Theta { theta1 :: L 25 n1, theta2 :: L 10 26 } deriving Show

predict :: (KnownNat n, KnownNat n1, n1 ~ (1+n)) => NNModel (1+n) -> R n -> Int
predict nnModel features =
  let (h, a3, z3, a2, z2, a1) = feedForward nnModel (row features)
  in (V.maxIndex . unwrap . unrow) h

-- m: number of training examples
-- n: number of features
feedForward :: (KnownNat m, KnownNat n, KnownNat n1, n1 ~ (1+n), 1<=m) => NNModel n1 -> L m n -> (L m 10, L 10 m, L 10 m, L 26 m, L 25 m, L n1 m)
feedForward nnModel x =
  let Theta theta1 theta2 = nnModel
      a1 = (addOnes . tr) x
      z2 = theta1 <> a1
      a2 = addOnes $ sigmoid z2
      z3 = theta2 <> a2
      a3 = sigmoid z3
      h = tr a3
  in (h, a3, z3, a2, z2, a1)

trainNetwork :: (KnownNat m, KnownNat n, KnownNat n1, n1 ~ (1+n), n ~ (n1-1), 1<=m, 1<=n, 1<=n1, 1<=(n1-1)) => R m -> L m n -> NNModel n1 -> NNModel n1
trainNetwork y x initModel =
  let  lambda = 1 -- regularisation
       function = cost x y lambda
       --tolerance = 1e-5
       --stop (Params prev) (Params cur) = abs (fst (function prev) - fst (function cur)) < tolerance
       stop (Params prev) (Params cur) = abs (fst (function cur)) < 0.1
       stopCond = StopWhen stop
       alpha = 0.01
       nnModel = model $ gradientDescent function stopCond alpha (Params initModel)
  in nnModel

listToVector :: KnownNat m => [Double] -> Maybe (R m)
listToVector lst = (create . LA.vector) lst :: KnownNat m1 => Maybe (R m1)

listOfListToMatrix :: (KnownNat m, KnownNat n, Num a) => [[Double]] -> Maybe (L m n)
listOfListToMatrix listOfList = (create . LA.fromRows . map LA.vector ) listOfList :: (KnownNat m1, KnownNat n1) => Maybe (L m1 n1)

-- returns label, feature vector
parseLine :: String -> (Double, [Double])
parseLine s =
  let ints = (reverse . read) ("[" ++ s ++ "]") :: [Double]
  in (head ints, reverse $ tail ints)

parseFile :: String -> ([Double], [[Double]])
parseFile s = unzip $ parseLine <$> lines s

-- x: nb of training examples * nb_of_features
-- y: label corresponding to training examples
-- theta1 : nb of rows in hidden layer * (nb_of_features +1)
-- theta2 : nb of rows in output layer * (nb_of_rows(theta1) +1)
cost :: (KnownNat m, KnownNat n, KnownNat n1, 1<=m, 1<=n, n1 ~ (1+n), n ~ (n1-1), 1<=n1, 1<=(n1-1)) => L m n -> R m -> Double -> NNModel n1 -> (Double, NNModel n1)
cost x y lambda nnModel =
  let (h, a3, z3, a2, z2, a1) = feedForward nnModel x
      yb = toBinMatrix y
      thet1 = theta1 nnModel
      jmat = yb * (log h) + (1-yb) * log (1 - h)
      m = fromIntegral $ size y
      regul = lambda / (2*m) * regularizeNNModel nnModel
      j = - (sumElements jmat) / m
      grad = nnModel --backpropagate yb lambda nnModel z2 a1 a2 a3
  in (j + regul, grad)

regularizeNNModel :: (KnownNat n1, KnownNat n, 1<=n1, 1<=n, n ~ (n1-1)) => NNModel n1 -> Double
regularizeNNModel (Theta t1 t2) =
  let t1' = dropFirstColumn t1
      t2' = dropFirstColumn t2
      t1Sq = t1' * t1' -- TODO t1' ^2 works in the REPL but not here !?
      t2Sq = t2' * t2' -- TODO t1' ^2 works in the REPL but not here !?
  in sumElements t1Sq + sumElements t2Sq

sumElements :: (KnownNat m, KnownNat n) => L m n -> Double
sumElements = LA.sumElements . extract
  
dropFirstColumn :: forall m n n_1 . (KnownNat m, KnownNat n, KnownNat n_1, 1<=n, n_1 ~ (n-1)) => L m n -> L m n_1
dropFirstColumn m =
  let transposedM = (snd . headTailMatrix . tr) m
  in  tr transposedM
  
headTailMatrix :: forall m n . (KnownNat m, KnownNat n, 1<=m) => L m n -> (L 1 n, L (m-1) n)
headTailMatrix = splitRows

backpropagate :: (KnownNat m, KnownNat n1) => L m 10 -> Double -> NNModel n1 -> L 25 m -> L n1 m -> L 26 m -> L 10 m -> NNModel n1
backpropagate yb lambda (Theta theta1 theta2) z2 a1 a2 a3 =
  let delta3 = a3 - tr yb
      delta2a = tr theta2 <> delta3
      delta2b = snd (headTailMatrix delta2a)
      delta2 = delta2b * sigmoidGradient z2
      d2 = delta3 <> tr a2
      d1 = delta2 <> tr a1
      m = (fromIntegral . fst . size) yb
      gradTheta1NonReg  = (d1 / konst m)
      gradTheta2NonReg  = (d2 / konst m)
--      regulCoef = lambda / m
--      gradTheta1 = (d1 / konst m) + (konst regulCoef * theta1) -- TODO not for the first column
  in Theta gradTheta1NonReg gradTheta2NonReg  -- TODO regularization

traceMatrix :: (KnownNat m, KnownNat n, 1<=m) => String -> L m n -> L m n
traceMatrix s m =
  let line1 = show $ (extract . fst . headTailMatrix) m
  in  trace (s ++ ": " ++ line1) m
 

instance (KnownNat n1) => GradientDescent (NNModel n1 -> (Double, NNModel n1)) where
  data Params (NNModel n1 -> (Double, NNModel n1)) = Params { model :: NNModel n1 }


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
