library(StatsBombR)
library(dplyr)
library(tidyverse)
library(xgboost)
library(caret)
library(vip)
library(gt)
library(gtExtras)

comps <- FreeCompetitions() %>%
  filter(competition_id == 43 & season_id == 106)

matches <- FreeMatches(comps)

events <- free_allevents(matches, Parallel = T)
events <- allclean(events) 

goals <- events %>%
  filter(shot.outcome.name == "Goal")

goals <- shotinfo(goals)

proc_events <- events %>%
  mutate(pos_id = paste0(match_id, "-", possession), ball_receipt = ifelse(type.name == "Ball Receipt*" & is.na(ball_receipt.outcome.name), 1, 0)) %>%
  group_by(pos_id) %>%
  mutate(prev_ball_receipts = cumsum(ball_receipt) - ball_receipt) %>%
  ungroup() %>%
  group_by(match_id) %>%
  mutate(under_pressure = ifelse(is.na(under_pressure), 0, under_pressure), counterpress = ifelse(is.na(counterpress), 0, counterpress), under_pressure_prev = lag(under_pressure), counterpress_prev = lag(counterpress)) %>%
  ungroup() 

select_events <- proc_events %>%
  select(index, match_id, type.name, possession_team.id, team.id, tactics.formation) 
  
formations <- select_events %>%
  group_by(match_id, team.id) %>%
  fill(tactics.formation, .direction = "downup")

final_formation <- formations %>%
  group_by(match_id) %>%
  mutate(team_num = as.character(dense_rank(team.id)), keep_team_num = team_num) %>%
  pivot_wider(names_from = team_num, values_from = tactics.formation, 
              names_prefix = "team_") %>%
  fill(team_1, .direction = "downup") %>%
  fill(team_2, .direction = "downup") %>%
  mutate(formation = case_when((keep_team_num == 1 & possession_team.id == team.id) ~ team_1,
                               (keep_team_num == 1 & possession_team.id != team.id) ~ team_2,
                               (keep_team_num == 2 & possession_team.id == team.id) ~ team_2,
                               (keep_team_num == 2 & possession_team.id != team.id) ~ team_1)) %>%
  select(index, match_id, formation)

final_events <- left_join(proc_events, final_formation, by = c("index", "match_id")) 

data <- final_events %>%
  filter(type.name == "Ball Receipt*") %>%
  mutate(time = 60 * minute + second, formation = as.factor(formation), position = as.factor(position.name), location = gsub("c\\(|\\)", "", location)) %>%
  separate(location, into = c("x", "y"), sep = ",", convert = TRUE) %>%
  mutate(dist_goal = sqrt(x^2 + (y-40)^2)) %>%
  group_by(match_id, possession_team.id) %>%
  select(name = player.name, id = player.id, match_id, dist_goal, position, time, prev_ball_receipts, formation, under_pressure_prev, counterpress_prev)

group_stage_matches <- matches %>%
  mutate(group_stage = ifelse(competition_stage.name == "Group Stage", 1, 0)) %>%
  select(match_id, group_stage)

final_data <- left_join(data, group_stage_matches, by = "match_id")

factor_data <- final_data %>%
  ungroup() %>%
  select(position, formation)

dummy <- dummyVars(" ~ .", data = factor_data)
factors <- data.frame(predict(dummy, newdata = factor_data))

final_data <- cbind(final_data, factors) %>%
  select(-position, -formation)

xgboost_train <- final_data %>%
  filter(group_stage == 1)

xgboost_test <- final_data %>%
  filter(group_stage != 1)

labels_train <- as.matrix(xgboost_train[, 5])
xgboost_trainfinal <- as.matrix(xgboost_train[, c(6:9, 11:44)])
xgboost_testfinal <- as.matrix(xgboost_test[, c(6:9, 11:44)])

doe_model <- xgboost(data = xgboost_trainfinal, label = labels_train, nrounds = 100, objective = "reg:squarederror", early_stopping_rounds = 10, max_depth = 6, eta = 0.3)

vip(doe_model)

dist_predicted <- predict(doe_model, xgboost_testfinal)
dist <- as.matrix(xgboost_test[,5])
postResample(dist_predicted, dist)

dist_predictions <- as.data.frame(
  matrix(predict(doe_model, as.matrix(final_data[,c(6:9, 11:44)])))
)

all_stats <- cbind(final_data, dist_predictions) %>%
  select(name, team = possession_team.id, group_stage, id, dist_goal, pred_dist_goal = V1)

all_stats <- all_stats %>%
  filter(group_stage == 0) %>%
  group_by(id) %>%
  summarize(ball_receipts = n(), name = first(name), team_id = as.integer(names(which.max(table(team)))), avg_dist_goal = mean(dist_goal), pred_avg_dist_goal = mean(pred_dist_goal), doe = avg_dist_goal - pred_avg_dist_goal) %>%
  filter(ball_receipts >= 50)

team_ids <- matches %>%
  select(team_id = home_team.home_team_id, team = home_team.home_team_name)

all_stats <- left_join(all_stats, team_ids, by = "team_id") %>%
  distinct(id, .keep_all = TRUE)

flags <- read_csv("flags_iso.csv") 

flags$Country[which(flags$`Alpha-2 code` == "NL")] <- "Netherlands"
flags$Country[which(flags$`Alpha-2 code` == "KR")] <- "South Korea"
flags$Country[which(flags$`Alpha-2 code` == "US")] <- "United States"

flags <- flags %>%
  select(team = Country, url = URL)

final_stats <- left_join(all_stats, flags, by = "team")

final_stats$url[which(final_stats$team == "England")] <- "https://cdn.britannica.com/44/344-004-494CC2E8/Flag-England.jpg"

top10 <- final_stats %>%
  arrange(-doe) %>%
  head(10) %>%
  select(name, team = url, doe)

bot10 <- final_stats %>%
  arrange(doe) %>%
  head(10) %>%
  select(name, team = url, doe)

t10 <- top10 %>% gt() %>% 
  gt_img_rows(columns = team) %>%
  gt_theme_538() %>%
  cols_align(
    align = "center",
    columns = c(name, team, doe)
  ) %>%
  gt_hulk_col_numeric(doe) %>%
  cols_label(
    name = md("**Player**"),
    team = md("**Team**"),
    doe = md("**DOE (Yards)**")
  ) %>%
  tab_header(
    title = md("**2022 FIFA World Cup Knockout Top 10 DOE (Average Distance from GK Over Expected)**"),
    subtitle = "Trained Data From 2022 FIFA World Cup Group Stage"
  ) %>%
  opt_align_table_header(align = "center")


gtsave(t10, "t10.png")

b10 <- bot10 %>% gt() %>% 
  gt_img_rows(columns = team) %>%
  gt_theme_538() %>%
  cols_align(
    align = "center",
    columns = c(name, team, doe)
  ) %>%
  gt_hulk_col_numeric(doe, reverse = TRUE) %>%
  cols_label(
    name = md("**Player**"),
    team = md("**Team**"),
    doe = md("**DOE (Yards)**")
  ) %>%
  tab_header(
    title = md("**2022 FIFA World Cup Knockout Bottom 10 DOE (Average Distance from GK Over Expected)**"),
    subtitle = "Trained Data From 2022 FIFA World Cup Group Stage"
  ) %>%
  opt_align_table_header(align = "center")



gtsave(b10, "b10.png")
