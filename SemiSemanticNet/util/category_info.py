category_info = {
  'Chair': {
    'n_part_list': [6, 30, 39],
    'n_test': 1217,
    'n_su_list': {
      '0.02': 90,
      '0.05': 224,
      '0.1': 449,
      '0.2': 898
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Chair_ratio02_90_aug4489_un4489_17956.tfrecords',
      '0.05': 'data/PartNet/Chair_ratio05_224_aug4489_un4489_17956.tfrecords',
      '0.1': 'data/PartNet/Chair_ratio10_449_aug4489_un4489_17956.tfrecords',
      '0.2': 'data/PartNet/Chair_ratio20_898_aug4489_un4489_17956.tfrecords',
    },
    'test_dataset': 'data/PartNet/Chair_level123_test_1217.tfrecords'
  },
  'Table': {
    'n_part_list': [11, 42, 51],
    'n_test': 1668,
    'n_su_list': {
      '0.02': 114,
      '0.05': 285,
      '0.1': 571,
      '0.2': 1141
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Table_ratio02_114_aug5707_un5707_22828.tfrecords',
      '0.05': 'data/PartNet/Table_ratio05_285_aug5707_un5707_22828.tfrecords',
      '0.1': 'data/PartNet/Table_ratio10_571_aug5707_un5707_22828.tfrecords',
      '0.2': 'data/PartNet/Table_ratio20_1141_aug5707_un5707_22828.tfrecords',
    },
    'test_dataset': 'data/PartNet/Table_level123_test_1668.tfrecords'
  },
  'Lamp': {
    'n_part_list': [18, 28, 41],
    'n_test': 419,
    'n_su_list': {
      '0.02': 31,
      '0.05': 78,
      '0.1': 155,
      '0.2': 311
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Lamp_ratio02_31_aug1554_un1554_6216.tfrecords',
      '0.05': 'data/PartNet/Lamp_ratio05_78_aug1554_un1554_6216.tfrecords',
      '0.1': 'data/PartNet/Lamp_ratio10_155_aug1554_un1554_6216.tfrecords',
      '0.2': 'data/PartNet/Lamp_ratio20_311_aug1554_un1554_6216.tfrecords',
    },
    'test_dataset': 'data/PartNet/Lamp_level123_test_419.tfrecords'
  },
  'StorageFurniture': {
    'n_part_list': [7, 19, 24],
    'n_test': 451,
    'n_su_list': {
      '0.02': 32,
      '0.05': 79,
      '0.1': 159,
      '0.2': 318
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/StorageFurniture_ratio02_32_aug1588_un1588_6352.tfrecords',
      '0.05': 'data/PartNet/StorageFurniture_ratio05_79_aug1588_un1588_6352.tfrecords',
      '0.1': 'data/PartNet/StorageFurniture_ratio10_159_aug1588_un1588_6352.tfrecords',
      '0.2': 'data/PartNet/StorageFurniture_ratio20_318_aug1588_un1588_6352.tfrecords',
    },
    'test_dataset': 'data/PartNet/StorageFurniture_level123_test_451.tfrecords'
  },
  'Bag': {
    'n_part_list': [4, 4, 4],
    'n_test': 29,
    'n_su_list': {
      '0.02': 2,
      '0.05': 5,
      '0.1': 9,
      '0.2': 18
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Bag_ratio02_2_aug92_un92_368.tfrecords',
      '0.05': 'data/PartNet/Bag_ratio05_5_aug92_un92_368.tfrecords',
      '0.1': 'data/PartNet/Bag_ratio10_9_aug92_un92_368.tfrecords',
      '0.2': 'data/PartNet/Bag_ratio20_18_aug92_un92_368.tfrecords',
    },
    'test_dataset': 'data/PartNet/Bag_level123_test_29.tfrecords'
  },
  'Bed': {
    'n_part_list': [4, 10, 15],
    'n_test': 37,
    'n_su_list': {
      '0.02': 3,
      '0.05': 7,
      '0.1': 13,
      '0.2': 27
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Bed_ratio02_3_aug133_un133_532.tfrecords',
      '0.05': 'data/PartNet/Bed_ratio05_7_aug133_un133_532.tfrecords',
      '0.1': 'data/PartNet/Bed_ratio10_13_aug133_un133_532.tfrecords',
      '0.2': 'data/PartNet/Bed_ratio20_27_aug133_un133_532.tfrecords',
    },
    'test_dataset': 'data/PartNet/Bed_level123_test_37.tfrecords'
  },
  'Bottle': {
    'n_part_list': [6, 6, 9],
    'n_test': 84,
    'n_su_list': {
      '0.02': 6,
      '0.05': 16,
      '0.1': 32,
      '0.2': 63
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Bottle_ratio02_6_aug315_un315_1260.tfrecords',
      '0.05': 'data/PartNet/Bottle_ratio05_16_aug315_un315_1260.tfrecords',
      '0.1': 'data/PartNet/Bottle_ratio10_32_aug315_un315_1260.tfrecords',
      '0.2': 'data/PartNet/Bottle_ratio20_63_aug315_un315_1260.tfrecords',
    },
    'test_dataset': 'data/PartNet/Bottle_level123_test_84.tfrecords'
  },
  'Bowl': {
    'n_part_list': [4, 4, 4],
    'n_test': 39,
    'n_su_list': {
      '0.02': 3,
      '0.05': 7,
      '0.1': 13,
      '0.2': 26
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Bowl_ratio02_3_aug131_un131_524.tfrecords',
      '0.05': 'data/PartNet/Bowl_ratio05_7_aug131_un131_524.tfrecords',
      '0.1': 'data/PartNet/Bowl_ratio10_13_aug131_un131_524.tfrecords',
      '0.2': 'data/PartNet/Bowl_ratio20_26_aug131_un131_524.tfrecords',
    },
    'test_dataset': 'data/PartNet/Bowl_level123_test_39.tfrecords'
  },
  'Clock': {
    'n_part_list': [6, 6, 11],
    'n_test': 98,
    'n_su_list': {
      '0.02': 8,
      '0.05': 20,
      '0.1': 41,
      '0.2': 81
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Clock_ratio02_8_aug406_un406_1624.tfrecords',
      '0.05': 'data/PartNet/Clock_ratio05_20_aug406_un406_1624.tfrecords',
      '0.1': 'data/PartNet/Clock_ratio10_41_aug406_un406_1624.tfrecords',
      '0.2': 'data/PartNet/Clock_ratio20_81_aug406_un406_1624.tfrecords',
    },
    'test_dataset': 'data/PartNet/Clock_level123_test_98.tfrecords'
  },
  'Dishwasher': {
    'n_part_list': [3, 5, 7],
    'n_test': 51,
    'n_su_list': {
      '0.02': 2,
      '0.05': 6,
      '0.1': 11,
      '0.2': 22
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Dishwasher_ratio02_2_aug111_un111_444.tfrecords',
      '0.05': 'data/PartNet/Dishwasher_ratio05_6_aug111_un111_444.tfrecords',
      '0.1': 'data/PartNet/Dishwasher_ratio10_11_aug111_un111_444.tfrecords',
      '0.2': 'data/PartNet/Dishwasher_ratio20_22_aug111_un111_444.tfrecords',
    },
    'test_dataset': 'data/PartNet/Dishwasher_level123_test_51.tfrecords'
  },
  'Display': {
    'n_part_list': [3, 3, 4],
    'n_test': 191,
    'n_su_list': {
      '0.02': 13,
      '0.05': 32,
      '0.1': 63,
      '0.2': 127
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Display_ratio02_13_aug633_un633_2532.tfrecords',
      '0.05': 'data/PartNet/Display_ratio05_32_aug633_un633_2532.tfrecords',
      '0.1': 'data/PartNet/Display_ratio10_63_aug633_un633_2532.tfrecords',
      '0.2': 'data/PartNet/Display_ratio20_127_aug633_un633_2532.tfrecords',
    },
    'test_dataset': 'data/PartNet/Display_level123_test_191.tfrecords'
  },
  'Door': {
    'n_part_list': [3, 4, 5],
    'n_test': 51,
    'n_su_list': {
      '0.02': 3,
      '0.05': 7,
      '0.1': 15,
      '0.2': 30
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Door_ratio02_3_aug149_un149_596.tfrecords',
      '0.05': 'data/PartNet/Door_ratio05_7_aug149_un149_596.tfrecords',
      '0.1': 'data/PartNet/Door_ratio10_15_aug149_un149_596.tfrecords',
      '0.2': 'data/PartNet/Door_ratio20_30_aug149_un149_596.tfrecords',
    },
    'test_dataset': 'data/PartNet/Door_level123_test_51.tfrecords'
  },
  'Earphone': {
    'n_part_list': [6, 6, 10],
    'n_test': 53,
    'n_su_list': {
      '0.02': 3,
      '0.05': 7,
      '0.1': 15,
      '0.2': 29
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Earphone_ratio02_3_aug147_un147_588.tfrecords',
      '0.05': 'data/PartNet/Earphone_ratio05_7_aug147_un147_588.tfrecords',
      '0.1': 'data/PartNet/Earphone_ratio10_15_aug147_un147_588.tfrecords',
      '0.2': 'data/PartNet/Earphone_ratio20_29_aug147_un147_588.tfrecords',
    },
    'test_dataset': 'data/PartNet/Earphone_level123_test_53.tfrecords'
  },
  'Faucet': {
    'n_part_list': [8, 8, 12],
    'n_test': 132,
    'n_su_list': {
      '0.02': 9,
      '0.05': 22,
      '0.1': 44,
      '0.2': 87
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Faucet_ratio02_9_aug435_un435_1740.tfrecords',
      '0.05': 'data/PartNet/Faucet_ratio05_22_aug435_un435_1740.tfrecords',
      '0.1': 'data/PartNet/Faucet_ratio10_44_aug435_un435_1740.tfrecords',
      '0.2': 'data/PartNet/Faucet_ratio20_87_aug435_un435_1740.tfrecords',
    },
    'test_dataset': 'data/PartNet/Faucet_level123_test_132.tfrecords'
  },
  'Hat': {
    'n_part_list': [6, 6, 6],
    'n_test': 45,
    'n_su_list': {
      '0.02': 3,
      '0.05': 8,
      '0.1': 17,
      '0.2': 34
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Hat_ratio02_3_aug170_un170_680.tfrecords',
      '0.05': 'data/PartNet/Hat_ratio05_8_aug170_un170_680.tfrecords',
      '0.1': 'data/PartNet/Hat_ratio10_17_aug170_un170_680.tfrecords',
      '0.2': 'data/PartNet/Hat_ratio20_34_aug170_un170_680.tfrecords',
    },
    'test_dataset': 'data/PartNet/Hat_level123_test_45.tfrecords'
  },
  'Keyboard': {
    'n_part_list': [3, 3, 3],
    'n_test': 31,
    'n_su_list': {
      '0.02': 2,
      '0.05': 6,
      '0.1': 11,
      '0.2': 22
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Keyboard_ratio02_2_aug111_un111_444.tfrecords',
      '0.05': 'data/PartNet/Keyboard_ratio05_6_aug111_un111_444.tfrecords',
      '0.1': 'data/PartNet/Keyboard_ratio10_11_aug111_un111_444.tfrecords',
      '0.2': 'data/PartNet/Keyboard_ratio20_22_aug111_un111_444.tfrecords',
    },
    'test_dataset': 'data/PartNet/Keyboard_level123_test_31.tfrecords'
  },
  'Knife': {
    'n_part_list': [5, 5, 10],
    'n_test': 77,
    'n_su_list': {
      '0.02': 4,
      '0.05': 11,
      '0.1': 22,
      '0.2': 44
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Knife_ratio02_4_aug221_un221_884.tfrecords',
      '0.05': 'data/PartNet/Knife_ratio05_11_aug221_un221_884.tfrecords',
      '0.1': 'data/PartNet/Knife_ratio10_22_aug221_un221_884.tfrecords',
      '0.2': 'data/PartNet/Knife_ratio20_44_aug221_un221_884.tfrecords',
    },
    'test_dataset': 'data/PartNet/Knife_level123_test_77.tfrecords'
  },
  'Laptop': {
    'n_part_list': [3, 3, 3],
    'n_test': 82,
    'n_su_list': {
      '0.02': 6,
      '0.05': 15,
      '0.1': 31,
      '0.2': 61
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Laptop_ratio02_6_aug306_un306_1224.tfrecords',
      '0.05': 'data/PartNet/Laptop_ratio05_15_aug306_un306_1224.tfrecords',
      '0.1': 'data/PartNet/Laptop_ratio10_31_aug306_un306_1224.tfrecords',
      '0.2': 'data/PartNet/Laptop_ratio20_61_aug306_un306_1224.tfrecords',
    },
    'test_dataset': 'data/PartNet/Laptop_level123_test_82.tfrecords'
  },
  'Microwave': {
    'n_part_list': [3, 5, 6],
    'n_test': 39,
    'n_su_list': {
      '0.02': 3,
      '0.05': 7,
      '0.1': 13,
      '0.2': 27
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Microwave_ratio02_3_aug133_un133_532.tfrecords',
      '0.05': 'data/PartNet/Microwave_ratio05_7_aug133_un133_532.tfrecords',
      '0.1': 'data/PartNet/Microwave_ratio10_13_aug133_un133_532.tfrecords',
      '0.2': 'data/PartNet/Microwave_ratio20_27_aug133_un133_532.tfrecords',
    },
    'test_dataset': 'data/PartNet/Microwave_level123_test_39.tfrecords'
  },
  'Mug': {
    'n_part_list': [4, 4, 4],
    'n_test': 35,
    'n_su_list': {
      '0.02': 3,
      '0.05': 7,
      '0.1': 14,
      '0.2': 28
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Mug_ratio02_3_aug138_un138_552.tfrecords',
      '0.05': 'data/PartNet/Mug_ratio05_7_aug138_un138_552.tfrecords',
      '0.1': 'data/PartNet/Mug_ratio10_14_aug138_un138_552.tfrecords',
      '0.2': 'data/PartNet/Mug_ratio20_28_aug138_un138_552.tfrecords',
    },
    'test_dataset': 'data/PartNet/Mug_level123_test_35.tfrecords'
  },
  'Refrigerator': {
    'n_part_list': [3, 6, 7],
    'n_test': 31,
    'n_su_list': {
      '0.02': 3,
      '0.05': 7,
      '0.1': 14,
      '0.2': 27
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Refrigerator_ratio02_3_aug136_un136_544.tfrecords',
      '0.05': 'data/PartNet/Refrigerator_ratio05_7_aug136_un136_544.tfrecords',
      '0.1': 'data/PartNet/Refrigerator_ratio10_14_aug136_un136_544.tfrecords',
      '0.2': 'data/PartNet/Refrigerator_ratio20_27_aug136_un136_544.tfrecords',
    },
    'test_dataset': 'data/PartNet/Refrigerator_level123_test_31.tfrecords'
  },
  'Scissors': {
    'n_part_list': [3, 3, 3],
    'n_test': 13,
    'n_su_list': {
      '0.02': 2,
      '0.05': 2,
      '0.1': 4,
      '0.2': 9
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Scissors_ratio02_2_aug45_un45_180.tfrecords',
      '0.05': 'data/PartNet/Scissors_ratio05_2_aug45_un45_180.tfrecords',
      '0.1': 'data/PartNet/Scissors_ratio10_4_aug45_un45_180.tfrecords',
      '0.2': 'data/PartNet/Scissors_ratio20_9_aug45_un45_180.tfrecords',
    },
    'test_dataset': 'data/PartNet/Scissors_level123_test_13.tfrecords'
  },
  'TrashCan': {
    'n_part_list': [5, 5, 11],
    'n_test': 63,
    'n_su_list': {
      '0.02': 4,
      '0.05': 11,
      '0.1': 22,
      '0.2': 44
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/TrashCan_ratio02_4_aug221_un221_884.tfrecords',
      '0.05': 'data/PartNet/TrashCan_ratio05_11_aug221_un221_884.tfrecords',
      '0.1': 'data/PartNet/TrashCan_ratio10_22_aug221_un221_884.tfrecords',
      '0.2': 'data/PartNet/TrashCan_ratio20_44_aug221_un221_884.tfrecords',
    },
    'test_dataset': 'data/PartNet/TrashCan_level123_test_63.tfrecords'
  },
  'Vase': {
    'n_part_list': [4, 4, 6],
    'n_test': 233,
    'n_su_list': {
      '0.02': 15,
      '0.05': 37,
      '0.1': 74,
      '0.2': 148
    },
    'train_dataset_list': {
      '0.02': 'data/PartNet/Vase_ratio02_15_aug741_un741_2964.tfrecords',
      '0.05': 'data/PartNet/Vase_ratio05_37_aug741_un741_2964.tfrecords',
      '0.1': 'data/PartNet/Vase_ratio10_74_aug741_un741_2964.tfrecords',
      '0.2': 'data/PartNet/Vase_ratio20_148_aug741_un741_2964.tfrecords',
    },
    'test_dataset': 'data/PartNet/Vase_level123_test_233.tfrecords'
  },
}
