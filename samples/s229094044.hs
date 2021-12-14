qq :: Int -> [String]
qq n = map (\(i,j) -> show i ++ "x" ++ show j ++ "=" ++ show (i*j))
    [(i,j) | i <- [1..n], j <- [1..n]]

main = mapM_ putStrLn $ qq 9