{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}

module HLearning.NeuralNetwork where

import GHC.TypeLits
import Numeric.LinearAlgebra.Static
import qualified Numeric.LinearAlgebra as LA
import qualified Data.Vector.Storable as V
import HLearning.Util



-- @return label, feature vector
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
      jmat = (-1 * yb) * log h
      -- JMat = ((-1*YB) .* log(H))  -  ((1-YB) .* log(1 - H));
      -- J = sum(sum(JMat)) / m;
      j = 3
  in j

--   -- read this: https://www.schoolofhaskell.com/user/konn/prove-your-haskell-for-great-safety/dependent-types-in-haskell
sigmoid :: (KnownNat m, KnownNat n) => L m n -> L m n
sigmoid a = 1 / (1 + exp(-a))

addOnes :: (KnownNat m, KnownNat n) => L m n -> L (1+m) n
addOnes x = konst 1 === x
--costTheta =
