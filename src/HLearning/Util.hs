{-# LANGUAGE DataKinds #-}
module HLearning.Util where



import GHC.TypeLits
import Numeric.LinearAlgebra.Static
import qualified Numeric.LinearAlgebra as LA
import qualified Data.Vector.Storable as V
import Debug.Trace


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
