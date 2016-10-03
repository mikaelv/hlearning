{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module HLearning.GradientDescent where

import Debug.Trace


-- Define a way to decide when to stop.
-- This lets the user specify an error tolerance easily.
-- The function takes the previous two sets of parameters and returns
-- `True` to continue the descent and `False` to stop.
newtype StopCondition a = StopWhen (Params a -> Params a -> Bool)

stopCondition f tolerance = StopWhen stop
  where stop (Arg prev) (Arg cur) =
          abs (f prev - f cur) < tolerance



gradientDescent :: GradientDescent a => a  -- What to optimize.
                -> StopCondition a        -- When to stop.
                -> Double                 -- Step size (alpha).
                -> Params a               -- Initial point (x0).
                -> Params a               -- Return: Location of minimum.
gradientDescent function (StopWhen stop) alpha x0 =
  let iterations = iterate takeStep x0
      iterationPairs = zip iterations $ tail iterations
    in
      -- Drop all elements where the resulting parameters (and previous parameters)
      -- do not meet the stop condition. Then, return just the last parameter set.
      snd . head $ dropWhile (not . uncurry stop) iterationPairs
  where
    -- For each step...
    takeStep params =
      -- Compute the gradient.
      let gradients = grad function params in
        -- And move against the gradient with a step size alpha.
        paramMove (-alpha) gradients params


class GradientDescent a where
  -- Type to represent the parameter space.
  data Params a :: *

  -- Compute the gradient at a location in parameter space.
  grad :: a -> Params a -> Params a

  -- Move in parameter space.
  paramMove :: Double    -- Scaling factor.
            -> Params a  -- Direction vector.
            -> Params a  -- Original location.
            -> Params a  -- New location.



instance (Floating a, Show a) => GradientDescent (a -> a) where
  -- The parameter for a function is just its argument.
  data Params (a -> a) = Arg { unArg :: a }

  -- Use numeric differentiation for taking the gradient.
  grad f (Arg value) = Arg $ (f value - f (value - epsilon)) / epsilon
    where epsilon = 0.0001

  paramMove scale (Arg vec) (Arg old) =
    let newLocation = old + fromRational (toRational scale) * vec
    in traceShow newLocation $ Arg newLocation
