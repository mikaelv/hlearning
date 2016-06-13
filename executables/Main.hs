{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}

module Main(toBinMatrix, main) where



import GHC.TypeLits
import Numeric.LinearAlgebra.Static
import qualified Data.ByteString.Char8 as B
import qualified Numeric.LinearAlgebra as LA
import qualified Data.Vector.Storable as V


main = do
  training <- readTrainingData
  mapM putStrLn $ fmap show training

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

toBinMatrix :: (KnownNat n) => R n -> L n 10
toBinMatrix v =
  let lst = LA.toList $ extract v
      m = LA.fromRows $ fmap (toBinaryVector . round) lst
      Just staticM = create m
  in staticM

-- Given an Int i between 0 and 9, makes a vector with the i-th element set to 1.0
toBinaryVector :: Int -> LA.Vector Double
toBinaryVector i =
  let (before, after) = splitAt i [0,0,0,0,0,0,0,0,0,0]
      lst = before ++ [1.0] ++ tail after
  in LA.vector lst

sigmoid :: (KnownNat m, KnownNat n) => L m n -> L m n
sigmoid a = 1 / (1 + exp(-a))

addOnes :: (KnownNat m, KnownNat n) => L m n -> L (1+m) n
addOnes x = konst 1 === x
--costTheta =


--readTrainingData :: IO [(Integer, [Integer])]
readTrainingData =
  let file = readFile "/Users/mikael/Documents/boulotmik/projects/hlearning/data/optdigits.tra"
  in fmap parseFile file
