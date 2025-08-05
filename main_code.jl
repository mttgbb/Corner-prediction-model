# Load necessary packages
using DataFrames, XLSX, StatsBase, Optim, Distributions, CSV

# Load training and test datasets from Excel files (see README.md for sample test and training sets)
df = DataFrame(XLSX.readtable("data/train.xlsx", "train"))  # Replace with correct filepath and sheet name
df_test = DataFrame(XLSX.readtable("data/test.xlsx", "test"))  # Replace with correct filepath and sheet name

# Compute average number of home and away corners per league
league_corners_avg_df = combine(groupby(df, :LeagueId),
    :Home_Corners => mean => :Home_Corners,
    :Away_Corners => mean => :Away_Corners)

# Store league-level average total corners (home + away) in a dictionary
league_corners_avg = Dict()
for row in eachrow(league_corners_avg_df)
    league_corners_avg[row.LeagueId] = row.Home_Corners + row.Away_Corners
end

# List of all teams
teams = union(df.HomeTeamId, df.AwayTeamId)

# Generalized Poisson PMF (used in likelihood function)
function g_poisson_pmf(y, θ, μ)
	λ = μ * θ
	if y == 0
		return exp(-λ)
	else
		return exp(-λ) * sum((λ^k / factorial(big(k))) * binomial(y-1, k-1) * (θ^k) * ((1-θ)^(y-k)) for k in 1:y)
	end
end

# Negative log-likelihood function to be minimized
function neg_loglik(params)
    k, θ, c, A, w_1, w_2 = params  # Model parameters
	cornerELO = Dict(teamid => 0.0 for teamid in teams)
	datelastplayed = Dict(teamid => 0.0 for teamid in teams)
	convergencecounter = Dict(teamid => 0 for teamid in teams)

    loss = 0.0
    for row in eachrow(df)
        hometeam = row.HomeTeamId
        awayteam = row.AwayTeamId
		league = row.LeagueId

        # Predict total corners from league average and ELO difference
		R = w_1 * cornerELO[hometeam] + w_2 * cornerELO[awayteam]
		exp_corners = league_corners_avg[league] * exp(R)
		exa_corners = row.Home_Corners + row.Away_Corners
		corners_err = exa_corners - exp_corners

        # Time since last match and convergence count
		hometeamdayssincelastmatch = row.Date - datelastplayed[hometeam]
		awayteamdayssincelastmatch = row.Date - datelastplayed[awayteam]
		hometeamgamecount = convergencecounter[hometeam]
		awayteamgamecount = convergencecounter[awayteam]

        # Learning rate adjustment based on match count and rest days
		k_home = hometeamgamecount < 10 ? k+A : min(sqrt(k^2 + c^2 * hometeamdayssincelastmatch), k+A)
		k_away = awayteamgamecount < 10 ? k+A : min(sqrt(k^2 + c^2 * awayteamdayssincelastmatch), k+A)

        # Update ELOs
		cornerELO[hometeam] += k_home * corners_err
		cornerELO[awayteam] += k_away * corners_err

        # Bookkeeping
		convergencecounter[hometeam] += 1
		convergencecounter[awayteam] += 1
		datelastplayed[hometeam] = row.Date
		datelastplayed[awayteam] = row.Date

        # Add to log-likelihood if both teams have stable estimates
		if hometeamgamecount >= 10 && awayteamgamecount >= 10 &&
		   hometeamdayssincelastmatch < 50 && awayteamdayssincelastmatch < 50

			prob_corners_correct = g_poisson_pmf(Int(exa_corners), θ, exp_corners)
			loss += prob_corners_correct < 0 ? 1e6 : -log(prob_corners_correct)
		end
    end
    return loss
end

# Fit model parameters by minimizing the negative log-likelihood
initial_params = [0.24, 0.90, 0.3, 2.9, 0.02, 0.02]
result = optimize(neg_loglik, initial_params)
optimized_params = Optim.minimizer(result)

# Reconstruct the final ELO ratings after training
function ELOcalculator(params)
	k, θ, c, A, w_1, w_2 = params
	cornerELO = Dict(teamid => 0.0 for teamid in teams)
	datelastplayed = Dict(teamid => 0.0 for teamid in teams)
	convergencecounter = Dict(teamid => 0 for teamid in teams)

    for row in eachrow(df)
        hometeam = row.HomeTeamId
        awayteam = row.AwayTeamId
		league = row.LeagueId
		R = w_1 * cornerELO[hometeam] + w_2 * cornerELO[awayteam]
		exp_corners = league_corners_avg[league] * exp(R)
		exa_corners = row.Home_Corners + row.Away_Corners
		corners_err = exa_corners - exp_corners

		hometeamdayssincelastmatch = row.Date - datelastplayed[hometeam]
		awayteamdayssincelastmatch = row.Date - datelastplayed[awayteam]
		hometeamgamecount = convergencecounter[hometeam]
		awayteamgamecount = convergencecounter[awayteam]

		k_home = hometeamgamecount < 10 ? k+A : min(sqrt(k^2 + c^2 * hometeamdayssincelastmatch), k+A)
		k_away = awayteamgamecount < 10 ? k+A : min(sqrt(k^2 + c^2 * awayteamdayssincelastmatch), k+A)

		cornerELO[hometeam] += k_home * corners_err
		cornerELO[awayteam] += k_away * corners_err

		convergencecounter[hometeam] += 1
		convergencecounter[awayteam] += 1
		datelastplayed[hometeam] = row.Date
		datelastplayed[awayteam] = row.Date
    end
	return cornerELO, datelastplayed, convergencecounter
end

# Prepare ELO ratings and auxiliary dictionaries for prediction
preppedELO = ELOcalculator(optimized_params)
datelastplayed_rd = deepcopy(preppedELO[2])
convergencecounter_rd = deepcopy(preppedELO[3])

# Create DataFrame to track confidence in ELO ratings
ratings_dev_df = DataFrame((name => zeros(Int, nrow(df_test)) for name in [:maxdayssincelastmatch, :minconvergencecounter])...)
for (i, row) in enumerate(eachrow(df_test))
	hometeam = row.HomeTeamId
    awayteam = row.AwayTeamId
	ratings_dev_df[i, :].maxdayssincelastmatch = max(row.Date - datelastplayed_rd[hometeam], row.Date - datelastplayed_rd[awayteam])
	ratings_dev_df[i, :].minconvergencecounter = min(convergencecounter_rd[hometeam], convergencecounter_rd[awayteam])

    # Simulate the match so future predictions have correct context
	convergencecounter_rd[hometeam] += 1
	convergencecounter_rd[awayteam] += 1
	datelastplayed_rd[hometeam] = row.Date
	datelastplayed_rd[awayteam] = row.Date
end

# Generate samples from the generalized Poisson-Geometric compound distribution
function rand_geometric_poisson(μ, θ)
	λ = μ * θ
    N = rand(Poisson(λ))
	return N == 0 ? 0 : sum(rand(Geometric(θ)) + 1 for _ in 1:N)
end

# Simulate predictions and count outcomes relative to betting line
function predictor(params, cornerELO, datelastplayed, convergencecounter, N)
    s, k, θ, c, A, w = params
	count_df = DataFrame((name => zeros(Int, nrow(df_test)) for name in [:count_under, :count_at, :count_over])...)

	for _ in 1:N
		# Copy state so simulation doesn't affect next run
		cornerELO_j = deepcopy(cornerELO)
		datelastplayed_j = deepcopy(datelastplayed)
		convergencecounter_j = deepcopy(convergencecounter)
		
        for (i, row) in enumerate(eachrow(df_test))
            hometeam = row.HomeTeamId
            awayteam = row.AwayTeamId
			league = row.LeagueId
			R = cornerELO_j[hometeam] + w * cornerELO_j[awayteam]
			exp_corners = league_corners_avg[league] * exp(s * R)
			exa_corners = rand_geometric_poisson(exp_corners, θ)

			# Compare simulated corners to betting line
			if exa_corners < row.Line
				count_df[i, :count_under] += 1
			elseif exa_corners == row.Line
				count_df[i, :count_at] += 1
			else
				count_df[i, :count_over] += 1
			end

			# Update ratings with the simulated outcome
			corners_err = exa_corners - exp_corners
			hometeamdayssincelastmatch = row.Date - datelastplayed_j[hometeam]
			awayteamdayssincelastmatch = row.Date - datelastplayed_j[awayteam]
			hometeamgamecount = convergencecounter_j[hometeam]
			awayteamgamecount = convergencecounter_j[awayteam]

			k_home = hometeamgamecount < 10 ? k+A : min(sqrt(k^2 + c^2 * hometeamdayssincelastmatch), k+A)
			k_away = awayteamgamecount < 10 ? k+A : min(sqrt(k^2 + c^2 * awayteamdayssincelastmatch), k+A)

			cornerELO_j[hometeam] += k_home * corners_err
			cornerELO_j[awayteam] += k_away * corners_err

			convergencecounter_j[hometeam] += 1
			convergencecounter_j[awayteam] += 1
			datelastplayed_j[hometeam] = row.Date
			datelastplayed_j[awayteam] = row.Date
        end
	end
    return count_df
end

# Run Monte Carlo prediction with 1 million samples
count_results = predictor(optimized_params, preppedELO[1], preppedELO[2], preppedELO[3], 1_000_000)
for col in names(count_results)
    count_results[!, col] ./= 1_000_000  # Convert counts to probabilities
end

# Merge prediction outputs with original test set
df_test_merged = hcat(df_test, count_results, ratings_dev_df)
df_test_merged.f_over = zeros(Float64, nrow(df_test_merged))
df_test_merged.f_under = zeros(Float64, nrow(df_test_merged))

# Kelly betting function based on predicted edge
function kelly(p_under, p_at, p_over, under_price, over_price)
	if p_at !== 0
		p_over = p_over / (p_over + p_under)  # Redistribute probability if market offers push odds
	end
		
	f_over = max(0, p_over - (1 - p_over) / (over_price - 1))
	f_under = max(0, (1 - p_over) - p_over / (under_price - 1))
	return f_over, f_under
end

# Apply Kelly criterion to each test match
for row in eachrow(df_test_merged)
	f_over, f_under = kelly(row.count_under, row.count_at, row.count_over, row.Under, row.Over)
	row.f_over = f_over
	row.f_under = f_under
end

# Export final predictions to CSV
CSV.write("output.csv", df_test_merged)

# Note: We recommend only betting games which satisfy the training criteria:
# hometeamgamecount >= 10 && awayteamgamecount >= 10 &&
# hometeamdayssincelastmatch < 50 && awayteamdayssincelastmatch < 50
